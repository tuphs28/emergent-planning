import torch
import torch.nn as nn
import torch.utils
import torch.utils.data
from create_probe_dataset import ProbingDataset, ProbingDatasetCleaned
from typing import Optional
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
import time

class ConvProbe(nn.Module):
    def __init__(self, in_channels: int, out_dim: int, kernel_size: int, padding: int = 0, nl: bool = False):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_dim, kernel_size=kernel_size, padding=padding, bias=False)
        self.out_dim = out_dim
        self.loss_fnc = nn.CrossEntropyLoss()
    def forward(self, input: torch.tensor, targets: Optional[torch.tensor] = None):
        out = self.conv(input)
        if targets is not None:
            assert out.shape[0] == targets.shape[0]
            out = out.view(out.shape[0], self.out_dim, 64)
            targets = targets.view(out.shape[0], 64)
            loss = self.loss_fnc(out, targets)
        else:
            loss = None
        return out, loss
    
class LinProbe(nn.Module):
    def __init__(self, in_channels: int, out_dim: int, nl: bool = False):
        super().__init__()
        self.ff = nn.Linear(in_features=in_channels*64, out_features=out_dim*64, bias=False)
        self.out_dim = out_dim
        self.loss_fnc = nn.CrossEntropyLoss()
    def forward(self, input: torch.tensor, targets: Optional[torch.tensor] = None):
        input = input.view(input.shape[0], -1)
        out = self.ff(input)
        if targets is not None:
            assert out.shape[0] == targets.shape[0]
            out = out.view(out.shape[0], self.out_dim, 64)
            targets = targets.view(out.shape[0], 64)
            loss = self.loss_fnc(out, targets)
        else:
            loss = None
        return out, loss
    
if __name__ == "__main__":
    import pandas as pd
    import argparse
    import os
    
    parser = argparse.ArgumentParser(description="train square-level probes")
    parser.add_argument("--feature", type=str, default="agent_onto_after", help="square-level feature to probe for")
    parser.add_argument("--num_epochs", type=int, default=30, help="number of epochs to train for")
    parser.add_argument("--kernel", type=int, default=1, help="kernel size for local probes")
    parser.add_argument("--weight_decay", type=float, default=0.001, help="weight decay")
    parser.add_argument("--num_seeds", type=int, default=5, help="number of initialisation seeds")
    parser.add_argument("--model_name", type=str, default="250m", help="name of agent checkpoint on which to train probes")
    parser.add_argument("--convprobe_off", action="store_false", default=True, help="use linprobe rather than convprobe")
    parser.add_argument("--num_layers", type=int, default=3, help="number of convlstm layers the agent has")
    parser.add_argument('--resnet', action='store_true')
    args = parser.parse_args()

    channels = list(range(32))
    batch_size = 16
    
    if args.kernel == 1:
        padding = 0
    elif args.kernel == 3:
        padding = 1
    elif args.kernel == 5:
        padding = 2 
    elif args.kernel == 7:
        padding = 3
    else:
        raise ValueError("Kernel size not supported")


    if torch.cuda.is_available(): 
        device = torch.device("cuda")
    else: 
        device = torch.device("cpu") 
    
    
    layers = [(f"layer{k}", ((k*32) if args.resnet else (k*64)+32)) for k in range(args.num_layers)] + [("x", 0)]
    
    results = {}

    train_dataset_c = torch.load(f"./data/train_data_full_{args.model_name}" + ("_resnet" if args.resnet else "") + ".pt", weights_only=False)
    test_dataset_c = torch.load(f"./data/test_data_full_{args.model_name}" + ("_resnet" if args.resnet else "") + ".pt", weights_only=False)

    cleaned_train_data, cleaned_test_data = [], []
    for trans in train_dataset_c.data:
        if type(trans[args.feature]) == int:
            if trans[args.feature] != -1:
                cleaned_train_data.append(trans)
        else:
            cleaned_train_data.append(trans)
    for trans in test_dataset_c.data:
        if type(trans[args.feature]) == int:
            if trans[args.feature] != -1:
                cleaned_test_data.append(trans)
        else:
            cleaned_test_data.append(trans)
    train_dataset_c.data = cleaned_train_data
    test_dataset_c.data = cleaned_test_data
    out_dim = 1 + max([c[args.feature].max().item() for c in train_dataset_c.data])
    for seed in range(args.num_seeds): 
        print(f"=============== Seed: {seed} ================")
        torch.manual_seed(seed)
        
        cleaned_train_data = [(trans["hidden_states"].cpu(), trans["board_state"], trans[args.feature]) for trans in train_dataset_c.data]
        cleaned_test_data = [(trans["hidden_states"].cpu(), trans["board_state"], trans[args.feature]) for trans in test_dataset_c.data]
        train_dataset = ProbingDatasetCleaned(cleaned_train_data)
        test_dataset = ProbingDatasetCleaned(cleaned_test_data)
        
        for layer_name, layer_idx in layers:
    
            print(f"========= {layer_name=}, {layer_idx=} =========")
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True,persistent_workers=True)
            test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True,persistent_workers=True)

            if args.convprobe_off:
                probe = ConvProbe(in_channels=7 if layer_name=="x" else 32, out_dim=out_dim, kernel_size=args.kernel, padding=(0 if args.kernel==1 else 1))
            else:
                probe = LinProbe(in_channels=7 if layer_name=="x" else 32, out_dim=out_dim)
            probe.to(device)
            optimiser = torch.optim.AdamW(params=probe.parameters(), lr=1e-3, weight_decay=args.weight_decay)
        
            for epoch in range(1, args.num_epochs+1):
                start_time = time.time()
                for hiddens, states, targets in train_loader:
                    hiddens = states.to(torch.float).to(device) if layer_name=="x" else hiddens[:,-1,[layer_idx+c for c in channels],:,:].to(device)
                    targets = targets.to(torch.long).to(device)
                    optimiser.zero_grad()
                    logits, loss = probe(hiddens, targets)
                    loss.backward()
                    optimiser.step()
                full_acc = 0
                if epoch % 1 == 0:
                    with torch.no_grad():
                        labs, preds = [], []
                        for hiddens, states, targets in test_loader:
                            hiddens = states.to(torch.float).to(device) if layer_name=="x" else hiddens[:,-1,[layer_idx+c for c in channels],:,:].to(device)
                            targets = targets.to(torch.long).to(device)
                            logits, loss = probe(hiddens, targets)
                            full_acc += (torch.sum(logits.argmax(dim=1)==targets.view(-1,64)).item())
                            preds += logits.argmax(dim=1).view(-1).tolist()
                            labs += targets.view(-1).tolist()

                        #if out_dim == 2:
                        #    prec, rec, f1, sup = precision_recall_fscore_support(labs, preds, average='binary', pos_label=1, zero_division=1, labels=[0,1])
                        #else:
                        prec, rec, f1, sup = precision_recall_fscore_support(labs, preds, average='macro', zero_division=1, labels=list(range(out_dim)))
                        
                        precisions, recalls, fones, _ = precision_recall_fscore_support(labs, preds, average=None, zero_division=1, labels=list(range(out_dim)))

                        print(f"---- Epoch {epoch} -----")
                        print("Full acc:", full_acc/(len(test_dataset.data)*64))
                        print("F1:", f1)
                        print("Time:", time.time()-start_time)

            results_dict = {"Acc": full_acc/(len(test_dataset.data)*64)}
            for j in range(out_dim):
                results_dict[f"Precision_{j}"] = precisions[j]
                results_dict[f"Recall_{j}"] = recalls[j]
                results_dict[f"F1_{j}"] = fones[j]
            results_dict["Avg_F1"] = f1
            results[f"{layer_name}_hidden_states"] = results_dict

            if not os.path.exists("./results"):
                os.mkdir("./results")
            if not os.path.exists("./results/convprobe_results"):
                os.mkdir("./results/convprobe_results")
            if not os.path.exists("./results/convprobe_results/models"):
                os.mkdir("./results/convprobe_results/models")
            if not os.path.exists(f"./results/convprobe_results/models/{args.feature}"):
                os.mkdir(f"./results/convprobe_results/models/{args.feature}")
            torch.save(probe.state_dict(), f"./results/convprobe_results/models/{args.feature}/{args.model_name }_{layer_name}_kernel{args.kernel}_wd{args.weight_decay}_seed{seed}.pt")

        results_df = pd.DataFrame(results)
        results_df.to_csv(f"./results/convprobe_results/{args.model_name}_{args.feature}_kernel{args.kernel}_wd{args.weight_decay}_seed{seed}.csv")
