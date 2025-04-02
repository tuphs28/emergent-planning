import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import Sampler
import yaml
import argparse
import re
import gc

import torch
from torch import nn
from thinker import util
from thinker.core.file_writer import FileWriter
from thinker.core.module import ResBlock
from detect_train import CustomDataset, ChunkSampler

class Decoder(nn.Module):
    def __init__(
        self,
        decoder_depth,
        input_shape
    ):
        super(Decoder, self).__init__()
        self.input_shape = input_shape        
        self.decoder_depth = decoder_depth

        frame_channels, h, w = input_shape
        out_channels = 64
        n_block = 1

        d_conv = [
            ResBlock(inplanes=out_channels * 2, disable_bn=False)
            for _ in range(n_block)
        ]
        kernel_sizes = [4, 4, 4, 4]
        conv_channels = [
            frame_channels,
            out_channels,
            out_channels * 2,
            out_channels * 2,
            out_channels * 2,
        ]
        for i in range(self.decoder_depth, 4):
            if i in [1, 3]:
                d_conv.extend(
                    [ResBlock(inplanes=conv_channels[4 - i], disable_bn=False) for _ in range(n_block)]
                )
            d_conv.append(nn.ReLU())
            d_conv.append(
                nn.ConvTranspose2d(
                    conv_channels[4 - i],
                    conv_channels[4 - i - 1],
                    kernel_size=kernel_sizes[i],
                    stride=2,
                    padding=1,
                )
            )            
        self.d_conv = nn.Sequential(*d_conv)   

    def forward(self, x):
        return self.d_conv(x)
    

def transform_data(xs, device, flags):
    xs_ = {}

    env_state = xs["env_state"][:, 0]
    env_state = env_state.float()
    xs_["env_state"] = env_state.to(device)

    real_state = xs["real_state"][:, 0]
    real_state = real_state.float()
    xs_["real_state"] = real_state.to(device)
    return xs_

def train_epoch(decoder, dataloader, optimizer, device, flags, train=True):
    if train:
        decoder.train()
    else:
        decoder.eval()     
    running_train_eval = {}   
    with torch.set_grad_enabled(train):
        step = 0
        for xs, _ in dataloader:
            xs = transform_data(xs, device, flags)
            pred_real_state = decoder(xs["env_state"])
            loss = torch.mean(torch.square(pred_real_state - xs["real_state"]))
            train_eval = {}
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

def save_ckp(path, epoch, flags, optimizer, net):
    # save checkpoint
    d = {
        "epoch": epoch,
        "flags": flags,
        "optimizer_state_dict": optimizer.state_dict(),
        "net_state_dict": net.state_dict(),
    }
    torch.save(d, path)

def load_ckp(path, optimizer, net):
    checkpoint = torch.load(path)
    net.load_state_dict(checkpoint['net_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    flags = checkpoint.get('flags', None) 
    return epoch, flags

def decode_train(flags):
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
    decoder = Decoder(
        decoder_depth = flags.model_decoder_depth,
        input_shape = flags_data.env_state_shape
    )

    # load optimizer
    optimizer = torch.optim.Adam(
        decoder.parameters(), lr=flags.learning_rate, 
    )

    if flags.ckp:
        checkpoint = torch.load(flags.tckp_path, torch.device("cpu"))
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        decoder.load_state_dict(checkpoint["net_state_dict"])
        epoch = checkpoint["epoch"]
        del checkpoint
    else:
        epoch = 0

    decoder = decoder.to(device)
    util.optimizer_to(optimizer, device)

    print("Decoder network size: %d"
            % sum(p.numel() for p in decoder.parameters())
        )
    
    best_val_loss = float('inf')
    epoch_since_improve = 0

    stats = []
    while (epoch < flags.num_epochs):
        train_stat = train_epoch(decoder, dataloader, optimizer, device, flags, train=True)
        val_stat = train_epoch(decoder, val_dataloader, None, device, flags, train=False)
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
                save_ckp(flags.tckp_path_b, epoch, flags, optimizer, decoder)
                print(f"New best model saved to {flags.tckp_path_b}")
            else:
                epoch_since_improve += 1
            
            if epoch_since_improve > flags.early_stop_n:
                print(f"Stopping early at epoch {epoch} due to no improvement in validation loss for {flags.early_stop_n} consecutive epochs.")
                load_ckp(flags.tckp_path_b, optimizer, decoder)
                break  # Stop the training loop
 
        if epoch % 5 == 0 or epoch >= flags.num_epochs:
            save_ckp(flags.tckp_path, epoch, flags, optimizer, decoder)
            print(f"Checkpoint saved to {flags.tckp_path}")
    
    # testing performance
    del dataloader, dataset, val_dataloader, val_dataset
    test_dataset = CustomDataset(datadir=flags.datadir, transform=None, data_n=5000, prefix="test")
    test_dataloader = DataLoader(test_dataset, batch_size=flags.batch_size, shuffle=True)
    test_stat = train_epoch(decoder, test_dataloader, None, device, flags, train=False)    
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
    parser = argparse.ArgumentParser(description=f"Thinker decoder network training")
    # data setting
    parser.add_argument("--dxpid", default="", help="Data file name")
    parser.add_argument("--dproject", default="detect", help="Data project name.")
    parser.add_argument("--datadir", default="../data/transition/__dxpid__/", help="Data directory.")    
    parser.add_argument("--outdir", default="../data/decode_log/__dxpid__/", help="Data directory.")    
    parser.add_argument("--txpid", default="test", help="training xpid of the run.")  
    parser.add_argument("--project", default="detect_post", help="Project of the run.")
    parser.add_argument("--chunk_n", default=1, type=int, help="Number of chunks; 1 for no chunking.")
    # input setting
    parser.add_argument("--batch_size", default=128, type=int, help="Batch size in training.")
    parser.add_argument("--learning_rate", default=0.0001, type=float, help="Learning rate.")
    parser.add_argument("--num_epochs", default=50, type=int, help="Number of epoch.")
    parser.add_argument("--early_stop_n", default=-1, type=int, help="Earlying stopping; <0 for no early stopping.")
    parser.add_argument("--data_n", default=50000, type=int, help="Training data size.")
    # store checkpoint setting
    parser.add_argument("--ckp", action="store_true", help="Enable loading from checkpoint.")
    parser.add_argument("--use_wandb", action="store_true", help="Enable logging to wandb.")

    flags = parser.parse_args()    
    flags.datadir = flags.datadir.replace("__dproject__", flags.dproject)
    flags.datadir = flags.datadir.replace("__dxpid__", flags.dxpid)
    flags.outdir = flags.outdir.replace("__dproject__", flags.dproject)
    flags.outdir = flags.outdir.replace("__dxpid__", flags.dxpid)

    decode_train(flags)    