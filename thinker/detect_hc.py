import os
import numpy as np
import yaml
import argparse
import torch
from torch.utils.data import DataLoader
from torch.nn import functional as F
import torchvision.transforms as transforms
from PIL import Image
from detect_train import CustomDataset, ChunkSampler, evaluate_detect


def detect_hc(datadir, imgdir, data_n=1000, batch_size=512, search_rank=0, legacy=False):

    # datadir = "/home/scuk/RS/thinker/data/detect/v5_sok-32052928-0"
    dataset = CustomDataset(datadir=datadir, transform=None, chunk_n=1, data_n=data_n)
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=ChunkSampler(dataset))

    device = torch.device("cuda")

    # load setting
    yaml_file_path = os.path.join(datadir, 'config_detect.yaml')
    with open(yaml_file_path, 'r') as file:
        flags_data = yaml.safe_load(file)
    flags_data = argparse.Namespace(**flags_data)
    num_actions = flags_data.num_actions
    rec_t = flags_data.rec_t

    # Path to your BMP file
    # image_path = '/home/scuk/RS/thinker/data/player_on_dan_small.bmp'
    # Load the image
    image = Image.open(os.path.join(imgdir, "player_on_dan_small.bmp"))
    # Convert the image to a tensor
    transform = transforms.Compose([
        transforms.ToTensor(),  # Converts to Tensor, scales to [0, 1] range
    ])
    search_image = transform(image).to(device)

    def find_max_similarity_single_function(x, search_image):
        B, C, H, W = x.shape  # Batch size, Channels, Height, Width
        block_size = 8
        num_blocks_h = H // block_size  # Number of horizontal blocks
        num_blocks_w = W // block_size  # Number of vertical blocks

        x_reshaped = x.view(B, C, num_blocks_h, block_size, num_blocks_w, block_size)
        # Permute to group blocks together while keeping the batch and channel dimensions intact
        x_permuted = x_reshaped.permute(0, 2, 4, 1, 3, 5)
        # Flatten the block grid dimensions to list all blocks sequentially
        x_blocks = x_permuted.reshape(B, num_blocks_h * num_blocks_w, C * block_size * block_size)
        # Normalize the blocks and the search_image
        x_blocks_norm = F.normalize(x_blocks+1e-6, p=2, dim=-1)  # Normalize over channel dimension
        search_image_norm = F.normalize(torch.flatten(search_image), p=2, dim=-1)

        similarity = torch.sum(x_blocks_norm * search_image_norm, dim=-1)

        # Find the maximum similarity for each image in the batch
        max_similarity, _ = similarity.view(B, -1).max(dim=1)
        return max_similarity

    def mask_top_rank(x, rank):
        # args: x (tensor) of shape (B, N); rank (int)
        # return a mask that equals 1 if the element of each row is the rank largest element
        B, N = x.shape
        sorted_values, _ = x.sort(dim=1, descending=True)
        ties = (sorted_values[:, 1:] - sorted_values[:, :-1]) != 0
        cum_ties = torch.cumsum(ties, dim=-1)
        cum_ties = torch.concat([torch.zeros(B, 1, device=x.device), cum_ties], dim=-1)
        idx = torch.argmax((cum_ties == rank).float(), dim=1)
        not_found = torch.all(~(cum_ties == rank), dim=-1)
        rank_values = sorted_values[torch.arange(B, device=x.device), idx]
        mask = x == rank_values.unsqueeze(-1)
        mask[not_found] = False
        return mask

    #B = 2048
    #env_state = torch.stack([dataset[idx][0]["env_state"] for idx in range(B)]).to(device)
    #tree_rep = torch.stack([dataset[idx][0]["tree_rep"] for idx in range(B)]).to(device)
    #target_y = torch.stack([dataset[idx][1] for idx in range(B)]).to(device)

    eval_results = {}

    with torch.set_grad_enabled(False):

        for xs, target_y in dataloader:

            env_state = xs["env_state"].to(device)
            tree_rep = xs["tree_rep"].to(device)
            target_y = target_y.to(device)

            B, rec_t = env_state.shape[:2]
            max_sim = find_max_similarity_single_function(torch.flatten(env_state, 0, 1), search_image)
            max_sim = max_sim.view(B, rec_t)

            # compute last rollout return
            if not legacy:
                idx_reset = num_actions * 4 + 6 
                idx_rr = idx_reset + flags_data.rec_t + 1
            else:
                idx_reset =  num_actions * 5 + 6 + num_actions * 5 + 3
                idx_rr = 5 * num_actions + 4
                
            reset = tree_rep[:, :, idx_reset].bool()
            rollout_return = tree_rep[:, :, idx_rr]

            last_rollout_return = rollout_return.clone()
            r = last_rollout_return[:, -1].clone()
            for n in range(flags_data.rec_t-1, -1, -1):
                r[reset[:, n]] = last_rollout_return[reset[:, n], n]
                last_rollout_return[:, n] = r  

            search_mask = torch.zeros(B, rec_t, dtype=torch.bool, device=device)
            search_mask[:, 0] = 1

            for m in range(search_rank+1):
                search_mask = search_mask | mask_top_rank(last_rollout_return, m)

            max_sim[~search_mask] = 0
            max_sim = torch.max(max_sim, dim=-1)[0]
            pred_y = max_sim > 0.95
            
            result = evaluate_detect(target_y, pred_y)
            for k, v in result.items():
                if k not in eval_results: 
                    eval_results[k] = [v]
                else:
                    eval_results[k].append(v)


    for k in eval_results:
        eval_results[k] = np.mean(np.array(eval_results[k]))

    print(eval_results)    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=f"Thinker handcraft detection")
    parser.add_argument("--dxpid", default="", help="Data file name")
    parser.add_argument("--dproject", default="detect", help="Data project name.")
    parser.add_argument("--datadir", default="../data/transition/__dxpid__/", help="Data directory.") 
    parser.add_argument("--imgdir", default="../data/", help="Data directory.") 
    parser.add_argument("--rank", default=0, type=int, help="Top number of rollout used in searching; 0 for only using the top; 1 for using the top 2, etc.")
    parser.add_argument("--legacy", action="store_true", help="Whether legacy tree rep is used.")

    flags = parser.parse_args()    
    flags.datadir = flags.datadir.replace("__dproject__", flags.dproject)
    flags.datadir = flags.datadir.replace("__dxpid__", flags.dxpid)

    print(f"Applying handcraft net on {flags.datadir} with rank {flags.rank}..")

    detect_hc(datadir=flags.datadir, imgdir=flags.imgdir, search_rank=flags.rank, legacy=flags.legacy)