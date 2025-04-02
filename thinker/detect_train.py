import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import Sampler
import yaml
import argparse
import re
import gc

import gym
import math
import torch
from torch import nn
from torch.nn import functional as F
from thinker.model_net import BaseNet, FrameEncoder
from thinker.legacy import ShallowAFrameEncoder
from thinker import util
from thinker.core.file_writer import FileWriter

class CustomDataset(Dataset):
    def __init__(self, datadir, transform=None, data_n=None, prefix="data", chunk_n=1):
        self.datadir = datadir
        self.transform = transform
        self.data = []  # Current chunk of data
        self.data_n = data_n
        self.prefix = prefix
        self.file_list = [f for f in os.listdir(datadir) if f.endswith('.pt') and f.startswith(self.prefix)]
        self.file_list = sorted(self.file_list, key=lambda x: int(re.search(r'\d+', x).group()) if re.search(r'\d+', x) else 0)
        
        xs, ys = torch.load(os.path.join(self.datadir, self.file_list[0]))
        self.t = xs['env_state'].shape[0]
        self.b = xs['env_state'].shape[2]
        self.samples_per_file = self.t * self.b

        if data_n is not None:
            self.len_list = [self.samples_per_file for _ in range(data_n // self.samples_per_file)]
            if data_n % self.samples_per_file > 0: self.len_list.append(data_n % self.samples_per_file)
            self.file_list = self.file_list[:len(self.len_list)]
        else:
            self.len_list = [self.samples_per_file for _ in range(len(self.file_list))]
        
        self.chunk_n = chunk_n
        self.current_chunk = 0  # To track which chunk is currently loaded
        self.total_files = len(self.file_list)
        self.files_per_chunk = max(1, self.total_files // self.chunk_n)
        self.samples_per_chunk = self.files_per_chunk * self.samples_per_file

    def _load_chunk(self, chunk_index):
        # Determine file range for the current chunk
        start_file = chunk_index * self.files_per_chunk
        end_file = min(start_file + self.files_per_chunk, self.total_files)
        self.data = []  # Clear current data
        gc.collect()
        self.current_chunk = chunk_index
        for i in range(start_file, end_file):
            data_tmp = []
            file_name = self.file_list[i]
            print(f"Loading {file_name}")
            xs, ys = torch.load(os.path.join(self.datadir, file_name))
            xs.pop('step_status', None)
            xs.pop('done', None)
            
            for t_idx in range(self.t):
                for b_idx in range(self.b):
                    flattened_xs = {k: v[t_idx, :, b_idx] for k, v in xs.items()}
                    flattened_ys = {k: v[t_idx, b_idx] for k, v in ys.items()}
                    data_tmp.append((flattened_xs, flattened_ys))
                    if len(data_tmp) >= self.len_list[i]: 
                        break
        
            assert len(data_tmp) >= self.len_list[i], f"data {i} should have at least {self.len_list[i]} samples instead of {len(data_tmp)}"
            self.data.extend(data_tmp)

    def __len__(self):
        if self.data_n is not None: return self.data_n
        return self.samples_per_file * self.total_files

    def __getitem__(self, idx):
        # Calculate which chunk the idx falls into        
        chunk_index = idx // self.samples_per_chunk
        
        # If the requested idx is not in the current chunk, load the correct chunk
        if chunk_index != self.current_chunk or not self.data:
            self._load_chunk(chunk_index)
        
        # Adjust idx to the current chunk
        idx_within_chunk = idx % self.samples_per_chunk
        xs, ys = self.data[idx_within_chunk]
        if self.transform:
            xs = {k: self.transform(v) for k, v in xs.items()}
        return xs, ys

class ChunkSampler(Sampler):
    def __init__(self, dataset):
        self.chunk_n = dataset.chunk_n
        self.data_n = dataset.data_n
        self.samples_per_chunk = dataset.samples_per_chunk
        if self.samples_per_chunk * self.chunk_n < self.data_n:
            self.chunk_n += 1        

    def __iter__(self):
        # Generate a list of chunk indices
        chunk_indices = np.arange(self.chunk_n)
        # Shuffle the list of chunk indices to determine the order in which chunks are processed
        np.random.shuffle(chunk_indices)
        
        # For each chunk, generate and shuffle indices within that chunk, then yield them
        for chunk_idx in chunk_indices:
            start_idx = chunk_idx * self.samples_per_chunk
            end_idx = min(start_idx + self.samples_per_chunk, self.data_n)
            indices = np.arange(start_idx, end_idx)
            np.random.shuffle(indices)
            for idx in indices:
                yield idx

    def __len__(self):
        return self.data_n    

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, max_len: int = 500):
        super().__init__()

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Arguments:
            x: Tensor, shape ``[batch_size, seq_len, embedding_dim]``
        """
        x = x + self.pe[:,:x.size(1)]
        return x

class DetectFrameEncoder(nn.Module):
    def __init__(
        self,
        input_shape,     
        out_size=128,
        downscale=True,
        decoder_depth=0,
    ):  
        super(DetectFrameEncoder, self).__init__()
        self.out_size = out_size
        self.oned_input = len(input_shape) == 1
        self.encoder = FrameEncoder(prefix="se",
                                    dim_rep_actions=None,
                                    input_shape=input_shape,                             
                                    size_nn=1,             
                                    downscale_c=2,    
                                    downscale=downscale,
                                    concat_action=False)
        self.decoder_depth = decoder_depth

        if not self.oned_input:
            self.conv = []
            in_ch =  self.encoder.out_shape[0]
            for ch in [64]:
                self.conv.append(nn.ReLU())
                self.conv.append(nn.Conv2d(in_channels=in_ch,
                                        out_channels=ch,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1,))
                in_ch = ch
            self.conv = nn.Sequential(*self.conv)
            out_size = in_ch * self.encoder.out_shape[1] * self.encoder.out_shape[2]
        else:
            out_size = self.encoder.out_shape[0]
        self.fc = nn.Sequential(nn.Linear(out_size, self.out_size))       

    def forward(self, x):
        # x in shape of (B, C, H, W)
        out = self.encoder.forward_pre_mem(x, actions=None, depth=self.decoder_depth)
        if not self.oned_input:
            out = self.conv(out)
            out = torch.flatten(out, start_dim=1)
        out = self.fc(out)
        return out                                
        
class DetectNet(BaseNet):
    def __init__(
        self,
        env_state_shape,
        tree_rep_shape,
        hidden_state_shape,
        dim_actions,
        num_actions,
        tuple_actions,
        decoder_depth,
        delay_n,
        flags,
    ):    
        super(DetectNet, self).__init__()
        
        self.env_state_shape = env_state_shape # in (C, H, W) 
        self.tree_rep_shape = tree_rep_shape # in (C,) 
        self.see_tree_rep = self.tree_rep_shape is not None 

        self.hidden_state_shape = hidden_state_shape # in (inner_t, C, H, W)        
        self.see_hidden_state = self.hidden_state_shape is not None 
        if self.see_hidden_state:
            if len(hidden_state_shape) == 3:
                hidden_state_shape = [1,] + hidden_state_shape
                self.hidden_state_need_expand = True
            else:
                self.hidden_state_need_expand = False

        self.dim_actions = dim_actions
        self.num_actions = num_actions
        self.tuple_actions = tuple_actions
        self.delay_n = delay_n
        if not self.tuple_actions:
            self.action_space = gym.spaces.Discrete(num_actions)
        else:
            self.action_space = gym.spaces.Tuple((gym.spaces.Discrete(num_actions),)*self.dim_actions)
        self.dim_rep_actions = self.dim_actions if self.dim_actions > 1 else self.num_actions

        self.disable_thinker = flags.disable_thinker

        self.enc_out_size = 128  
        self.embed_size = self.enc_out_size + self.dim_rep_actions
        if not self.disable_thinker: self.embed_size += 1
        if self.see_tree_rep: self.embed_size += tree_rep_shape[0]

        tran_nhead = 8
        reminder = tran_nhead - (self.embed_size % tran_nhead)              
        self.enc_out_size += reminder
        self.embed_size += reminder
        
        self.pos_encoder = PositionalEncoding(self.embed_size)
        FrameEncoder = ShallowAFrameEncoder if flags.shallow_encode else DetectFrameEncoder
        self.true_x_encoder = FrameEncoder(input_shape=env_state_shape, out_size=self.enc_out_size, decoder_depth=decoder_depth)
        
        if self.see_hidden_state:
            self.h_encoder = FrameEncoder(input_shape=hidden_state_shape[1:], out_size=self.enc_out_size, downscale=False)   
        else:
            self.pred_x_encoder = FrameEncoder(input_shape=env_state_shape, out_size=self.enc_out_size, decoder_depth=decoder_depth)
        

        encoder_layer = nn.TransformerEncoderLayer(d_model=self.embed_size, 
                                                   nhead=tran_nhead, 
                                                   dim_feedforward=flags.tran_ff_n,
                                                   batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, flags.tran_layer_n)
        self.decoder_depth = decoder_depth

        self.pred_action = flags.pred_action
        if not flags.pred_action:
            self.classifier = nn.Linear(self.embed_size, 1)
        else:
            self.classifier = nn.Linear(self.embed_size, delay_n * num_actions)

        self.beta = nn.Parameter(torch.tensor(0.5), requires_grad=False) # portion of negative class

    def forward(self, env_state, tree_rep, hidden_state, action, reset):
        """
        Forward pass of detection nn
        Args:
            env_state: float Tensor in shape of (B, rec_t, C, H, W); true and predicted frame
            tree_rep: float Tensor in shape of (B, rec_t, C); model output
            hidden_state: float Tensor in shape of (B, rec_t, inner_t, C, H, W); model output
            action: uint Tensor in shape of (B, rec_t, dim_actions); action (real / imaginary)
            reset: bool Tensor in shape of  (B, rec_t); reset action
        Return:
            logit: float Tensor in shape of (B); logit of classifier output
            p: float Tensor in shape of (B); prob of classifier output
        """
        B, rec_t = env_state.shape[:2]
        
        action = util.encode_action(action, self.action_space, one_hot=False)        
        true_proc_x = self.true_x_encoder(env_state[:,0])
        true_proc_x = true_proc_x.view(B, self.enc_out_size).unsqueeze(1) # (B, 1, C)
        if not self.see_hidden_state:
            pred_proc_x = self.pred_x_encoder(
                torch.flatten(env_state[:,1:], 0, 1),
                                            )            
            pred_proc_x = pred_proc_x.view(B, rec_t - 1, self.enc_out_size)  # (B, rec_t - 1, C)
            proc_x = torch.concat([true_proc_x, pred_proc_x], dim=1) # (B, rec_t, C)      
        else:
            if self.hidden_state_need_expand:
                hidden_state = hidden_state.unsqueeze(2)
            proc_h = self.h_encoder(torch.flatten(hidden_state[:,0], 0, 1))
            proc_h = proc_h.view(B, -1, self.enc_out_size)  # (B, inner_t, C)
            proc_x = torch.concat([true_proc_x, proc_h], dim=1) # (B, 1 + inner_t, C)

        embed = [proc_x]
        if not self.see_hidden_state:
            embed.append(action)
            if not self.disable_thinker: embed.append(reset.unsqueeze(-1))
        else:
            action = torch.broadcast_to(action[:, [0], :], (B, proc_x.shape[1], self.dim_rep_actions))
            embed.append(action)
            if not self.disable_thinker: 
                reset = torch.broadcast_to(reset.unsqueeze(-1), (B, proc_x.shape[1], self.dim_rep_actions))
                embed.append(reset)

        if self.see_tree_rep: embed.append(tree_rep)    
        embed = torch.concat(embed, dim=2) # (B, rec_t, embed_size)
        embed_pos = self.pos_encoder(embed)
        out = self.transformer_encoder(embed_pos)
        if not self.pred_action:
            logit = self.classifier(out[:, -1, :]).view(B)
            p = torch.sigmoid(logit)
        else:
            logit = self.classifier(out[:, -1, :]).view(B, self.delay_n, self.num_actions)
            p = torch.softmax(logit, dim=-1)
        return logit, p

def transform_data(xs, device, flags):
    xs_ = {}

    env_state = xs["env_state"]
    env_state = env_state.float()
    xs_["env_state"] = env_state.to(device)

    xs_["tree_rep"] = xs["tree_rep"].to(device) if "tree_rep" in xs else None

    action = xs["pri_action"]
    if not flags.tuple_actions:
        action = action.unsqueeze(-1)
    xs_["action"] = action.to(device)

    xs_["reset"] = xs["reset_action"].to(device) if "reset_action" in xs else None
    xs_["hidden_state"] = xs["hidden_state"].to(device) if "hidden_state" in xs else None
    return xs_

def evaluate_detect(target_y, pred_y, mask, pred_action):
    if not pred_action:
        # Binarize the predictions
        pred_y_binarized = (pred_y > 0.5).float()
        target_y = target_y.float()

        # Compute the accuracy
        acc = torch.mean((pred_y_binarized == target_y).float()).item()
        
        # Compute the recall
        true_positives = (pred_y_binarized * target_y).sum().float()
        possible_positives = target_y.sum().float()
        rec = (true_positives / (possible_positives + 1e-6)).item()
        
        # Compute the precision
        predicted_positives = pred_y_binarized.sum().float()
        prec = (true_positives / (predicted_positives + 1e-6)).item()
        
        # Compute the F1 score
        f1 = 2 * (prec * rec) / (prec + rec + 1e-6)   

        neg_p = 1 - torch.mean(target_y.float()).item()

        return {
            "acc": acc,
            "rec": rec,
            "prec": prec,
            "f1": f1,
            "neg_p": neg_p,
            }
    else:
        _, pred_classes = torch.max(pred_y, dim=2) 
        B, T = target_y.shape
        stats = {}
        accs = torch.zeros(B, T, device=target_y.device)                
        for K in range(1, T + 1):            
            correct_predictions = (pred_classes[:, :K] == target_y[:, :K]) & mask[:, :K]
            
            # Count the number of correct predictions and valid predictions
            num_correct = correct_predictions.sum(dim=1).float()  # Convert to float for division
            num_valid = mask[:, :K].sum(dim=1).float()
            
            # Avoid division by zero for cases where num_valid is 0
            num_valid = torch.where(num_valid == 0, torch.ones_like(num_valid), num_valid)
            
            # Step 3: Calculate accuracy for the first K steps
            accs[:, K-1] = num_correct / num_valid
            stats["acc_%d"%K] = torch.mean(accs[:, K-1]).item()
            if K == T: stats["acc"] = torch.mean(accs[:, K-1]).item()
        return stats

def train_epoch(detect_net, dataloader, optimizer, device, flags, train=True):
    if train:
        detect_net.train()
    else:
        detect_net.eval()     
    running_train_eval = {}   
    with torch.set_grad_enabled(train):
        step = 0
        for xs, target_ys in dataloader:
            xs = transform_data(xs, device, flags)

            if not flags.pred_action:
                target_y = target_ys["cost"]
            else:
                target_y = target_ys["last_real_actions"]
                y_done = target_ys["last_dones"]
                acc_done = torch.zeros_like(y_done)
                T = acc_done.shape[1]
                for t in range(1, T):
                    acc_done[:, t] = torch.logical_or(acc_done[:, t - 1], y_done[:, t - 1])
                acc_done = acc_done.to(device)
                    
            target_y = target_y.to(device)

            if flags.mask_im_state:
                xs["env_state"] = xs["env_state"].clone()
                xs["env_state"][:, 1:] = 0.

            if flags.mask_hidden_state:
                xs["hidden_state"] = torch.zeros_like(xs["hidden_state"])

            logit, pred_y = detect_net(**xs)            
            if not flags.pred_action:            
                n_mean_y = torch.mean((~target_y).float()).item()
                detect_net.beta.data = 0.99 * detect_net.beta.data + (1 - 0.99) * n_mean_y
                detect_net.beta.data.clamp_(0.05, 0.95)
                weights = torch.where(target_y == 1, detect_net.beta.data, 1-detect_net.beta.data)
                loss = F.binary_cross_entropy_with_logits(logit, target_y.float(), weight=weights)
            else:
                loss = F.cross_entropy(torch.flatten(logit, 0, 1), torch.flatten(target_y, 0, 1), reduction="none")
                mask = (1 - torch.flatten(acc_done, 0, 1).float())
                loss = loss * mask
                loss = torch.sum(loss) / torch.sum(mask)

            train_eval = evaluate_detect(target_y, pred_y, ~acc_done, flags.pred_action)
            train_eval["loss"] = loss.item()

            if train:
                optimizer.zero_grad()  # Zero the gradients
                loss.backward()  # Backward pass: compute gradient of the loss with respect to model parameters
                optimizer.step()  # Perform a single optimization step (parameter update)
            
            for key in train_eval.keys():
                if key not in running_train_eval: 
                    running_train_eval[key] = train_eval[key]
                else:
                    running_train_eval[key] += train_eval[key]
            step += 1
    return {key: val / step for (key, val) in running_train_eval.items()}

def save_ckp(path, epoch, flags, optimizer, detect_net):
    # save checkpoint
    d = {
        "epoch": epoch,
        "flags": flags,
        "optimizer_state_dict": optimizer.state_dict(),
        "net_state_dict": detect_net.state_dict(),
    }
    torch.save(d, path)

def load_ckp(path, optimizer, detect_net):
    checkpoint = torch.load(path)
    detect_net.load_state_dict(checkpoint['net_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    flags = checkpoint.get('flags', None) 
    return epoch, flags

def detect_train(flags):
    project = flags.project
    flags.datadir = os.path.abspath(os.path.expanduser(flags.datadir))
    tdir = os.path.abspath(os.path.expanduser(flags.outdir))
    if not os.path.exists(tdir): os.makedirs(tdir, exist_ok=True)

    if not flags.ckp:        
        # create ckp dir
        xpid_n = 0
        while (True):
            xpid_ = flags.txpid if xpid_n == 0 else flags.txpid + f"_{xpid_n}"
            ckpdir = os.path.join(tdir, xpid_)
            xpid_n += 1
            if not os.path.exists(ckpdir):
                os.mkdir(ckpdir) 
                flags.txpid = xpid_
                break    
    else:
        ckpdir = os.path.join(tdir, flags.txpid)
    flags.tckpdir = ckpdir
    flags.tckp_path = os.path.join(ckpdir, "ckp_detect.tar")
    flags.tckp_path_b = os.path.join(ckpdir, "ckp_detect_best.tar")
    print(f"Checkpoint path: {flags.tckp_path}")

    # load data
    dataset = CustomDataset(datadir=flags.datadir, transform=None, data_n=flags.data_n, chunk_n=flags.chunk_n)
    dataloader = DataLoader(dataset, batch_size=flags.batch_size, sampler=ChunkSampler(dataset))

    val_dataset = CustomDataset(datadir=flags.datadir, transform=None, data_n=5000, prefix="val")
    val_dataloader = DataLoader(val_dataset, batch_size=flags.batch_size, shuffle=True)

    # load setting
    yaml_file_path = os.path.join(flags.datadir, 'config_detect.yaml')
    with open(yaml_file_path, 'r') as file:
        flags_data = yaml.safe_load(file)
    flags_data = argparse.Namespace(**flags_data)
    flags = argparse.Namespace(**{**vars(flags), **vars(flags_data)}) # merge the two flags

    plogger = FileWriter(
        xpid=flags.txpid,
        xp_args=flags.__dict__,
        rootdir=tdir,
        overwrite=not flags.ckp,
    )
    flags.full_xpid = flags.dxpid + "_" + flags.txpid
    flags.project = project
    if flags.use_wandb: wlogger = util.Wandb(flags)

    # initalize net
    device = torch.device("cuda")
    detect_net = DetectNet(
        env_state_shape = flags_data.env_state_shape,
        tree_rep_shape = getattr(flags_data, "tree_rep_shape", None) if flags.see_tree_rep else None,
        hidden_state_shape = getattr(flags_data, "hidden_state_shape", None) if flags.see_hidden_state else None,
        dim_actions = flags_data.dim_actions,
        num_actions = flags_data.num_actions,
        tuple_actions = flags_data.tuple_actions,
        delay_n = flags_data.delay_n,
        decoder_depth = getattr(flags_data, "model_decoder_depth", 0),
        flags = flags,        
    )

    # load optimizer
    optimizer = torch.optim.Adam(
        detect_net.parameters(), lr=flags.learning_rate, 
    )

    if flags.ckp:
        checkpoint = torch.load(flags.tckp_path, torch.device("cpu"))
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        detect_net.load_state_dict(checkpoint["net_state_dict"])
        epoch = checkpoint["epoch"]
        del checkpoint
    else:
        epoch = 0

    detect_net = detect_net.to(device)
    util.optimizer_to(optimizer, device)

    print("Detect network size: %d"
            % sum(p.numel() for p in detect_net.parameters())
        )
    
    best_val_loss = float('inf')
    epoch_since_improve = 0

    stats = []
    while (epoch < flags.num_epochs):
        train_stat = train_epoch(detect_net, dataloader, optimizer, device, flags, train=True)
        val_stat = train_epoch(detect_net, val_dataloader, None, device, flags, train=False)
        stat = {**train_stat, **{'val/' + key: value for key, value in val_stat.items()}}
        stat["epoch"] = epoch
        stats.append(stat)
        plogger.log(stat)
        if flags.use_wandb: wlogger.wandb.log(stat, step=stat['epoch'])
        
        epoch += 1    
        print_str = f'Epoch {epoch}/{flags.num_epochs},'
        for key in stat.keys(): 
            if 'val/' + key in stat.keys():
                print_str += f" {key}:{stat[key]:.4f} ({stat['val/'+key]:.4f})"
        print(print_str)   
             
        # Early stopping and best model saving logic
        if flags.early_stop_n >= 0:  # Check if early stopping is enabled
            current_val_loss = val_stat['loss']  # Assuming val_stat contains the validation loss
            if current_val_loss < best_val_loss:
                best_val_loss = current_val_loss
                epoch_since_improve = 0
                save_ckp(flags.tckp_path_b, epoch, flags, optimizer, detect_net)
                print(f"New best model saved to {flags.tckp_path_b}")
            else:
                epoch_since_improve += 1
            
            if epoch_since_improve > flags.early_stop_n:
                print(f"Stopping early at epoch {epoch} due to no improvement in validation loss for {flags.early_stop_n} consecutive epochs.")
                load_ckp(flags.tckp_path_b, optimizer, detect_net)
                break  # Stop the training loop
 
        if epoch % 5 == 0 or epoch >= flags.num_epochs:
            save_ckp(flags.tckp_path, epoch, flags, optimizer, detect_net)
            print(f"Checkpoint saved to {flags.tckp_path}")
    
    # testing performance
    del dataloader, dataset, val_dataloader, val_dataset
    test_dataset = CustomDataset(datadir=flags.datadir, transform=None, data_n=5000, prefix="test")
    test_dataloader = DataLoader(test_dataset, batch_size=flags.batch_size, shuffle=True)
    test_stat = train_epoch(detect_net, test_dataloader, None, device, flags, train=False)    
    stat = {'test/' + key: value for key, value in test_stat.items()}
    print_str = f'Test performance,'
    for key in stat.keys():  print_str += f" {key}:{stat[key]:.4f}"
    print(print_str)            
    stat['epoch'] = epoch + 1
    stats.append(stat)
    plogger.log(stat)
    if flags.use_wandb: wlogger.wandb.log(stat, step=stat['epoch'])
    
    if flags.use_wandb: wlogger.wandb.save(os.path.join(flags.tckpdir, "*"), flags.tckpdir)
    np.save(os.path.join(flags.tckpdir, 'stats.npy'), np.array(stats, dtype=object))                
    plogger.close()    
    if flags.use_wandb: wlogger.wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=f"Thinker detection network training")
    # data setting
    parser.add_argument("--dxpid", default="", help="Data file name")
    parser.add_argument("--dproject", default="detect", help="Data project name.")
    parser.add_argument("--datadir", default="../data/transition/__dxpid__/", help="Data directory.")    
    parser.add_argument("--outdir", default="../data/detect_log/__dxpid__/", help="Data directory.")    
    parser.add_argument("--txpid", default="test", help="training xpid of the run.")  
    parser.add_argument("--project", default="detect_post", help="Project of the run.")
    parser.add_argument("--chunk_n", default=1, type=int, help="Number of chunks; 1 for no chunking.")
    # input setting
    parser.add_argument("--see_tree_rep", action="store_true", help="See tree rep or not.")
    parser.add_argument("--see_hidden_state", action="store_true", help="See hidden_state instead of future states")    
    parser.add_argument("--mask_hidden_state", action="store_true", help="Whether masking hidden_state or not.")
    parser.add_argument("--mask_im_state", action="store_true", help="Whether masking future env state or not.")
    # target output
    parser.add_argument("--pred_action", action="store_true", help="Whether to predict action instead of danger label.")
    # training setting
    parser.add_argument("--batch_size", default=128, type=int, help="Batch size in training.")
    parser.add_argument("--learning_rate", default=0.0001, type=float, help="Learning rate.")
    parser.add_argument("--num_epochs", default=50, type=int, help="Number of epoch.")
    parser.add_argument("--early_stop_n", default=-1, type=int, help="Earlying stopping; <0 for no early stopping.")
    parser.add_argument("--data_n", default=50000, type=int, help="Training data size.")
    # store checkpoint setting
    parser.add_argument("--ckp", action="store_true", help="Enable loading from checkpoint.")
    parser.add_argument("--use_wandb", action="store_true", help="Enable logging to wandb.")
    # net setting
    parser.add_argument("--tran_layer_n", default=3, type=int, help="Transformer layer size.")
    parser.add_argument("--tran_ff_n", default=512, type=int, help="Transformer encoder size.")
    parser.add_argument("--shallow_encode", action="store_true", help="Using shallow encoding.")

    flags = parser.parse_args()    
    flags.datadir = flags.datadir.replace("__dproject__", flags.dproject)
    flags.datadir = flags.datadir.replace("__dxpid__", flags.dxpid)
    flags.outdir = flags.outdir.replace("__dproject__", flags.dproject)
    flags.outdir = flags.outdir.replace("__dxpid__", flags.dxpid)

    detect_train(flags)