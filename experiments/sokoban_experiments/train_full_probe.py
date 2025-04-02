import torch
import torch.nn.functional as F
import torch.nn as nn
from typing import Optional
from create_probe_dataset import ProbingDataset, ProbingDatasetCleaned
import time
from sklearn.metrics import precision_recall_fscore_support
import pandas as pd
import os
import argparse

class LinearProbe(nn.Module):
    """Linear probe for the DRC(3,3) agent"""
    
    def __init__(self, in_dim: int, out_dim: int, bias: bool = False):
        super().__init__()
        self.in_dims = in_dim        
        self.out_dim = out_dim
        self.loss_fnc = nn.CrossEntropyLoss()

        self.proj = nn.Linear(in_features=self.in_dims, out_features=self.out_dim, bias=bias)

    def forward(self, input: torch.tensor, targets: Optional[torch.tensor] = None):
        input = input.view(-1, self.in_dims)
        out = self.proj(input.view(-1, self.in_dims))
        if targets is not None:
            #print(out.shape)
            #print(targets.shape)
            assert out.shape[0] == targets.shape[0]
            out = out.view(-1, self.out_dim)
            loss = self.loss_fnc(out, targets)
        else:
            loss = None
        return out, loss
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="train convprobes")
    parser.add_argument("--feature", type=str, default="agent_onto_after")
    parser.add_argument("--num_epochs", type=int, default=30)
    parser.add_argument("--weight_decay", type=float, default=0.001)
    parser.add_argument("--num_seeds", type=int, default=5)
    parser.add_argument("--model_name", type=str, default="250m")
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument('--resnet', action='store_true')
    args = parser.parse_args()

    
    layers = [(f"layer{k}",  ((k*32) if args.resnet else (k*64)+32))  for k in range(args.num_layers)] + [("x", 0)]
    batch_size = 16
    channels = list(range(32))

    if torch.cuda.is_available(): 
        device = torch.device("cuda")
    else: 
        device = torch.device("cpu") 
  
    results = {}

    train_dataset_c = torch.load(f"./data/train_data_full_{args.model_name}" + ("_resnet" if args.resnet else "") + ".pt", weights_only=False)
    test_dataset_c = torch.load(f"./data/test_data_full_{args.model_name}" + ("_resnet" if args.resnet else "") + ".pt", weights_only=False)
    
    cleaned_train_data, cleaned_test_data  = [], []
    for trans in train_dataset_c.data:
        if type(trans[args.feature ]) == int:
            if trans[args.feature ] != -1:
                cleaned_train_data.append(trans)
        else:
            cleaned_train_data.append(trans)
    for trans in test_dataset_c.data:
        if type(trans[args.feature ]) == int:
            if trans[args.feature ] != -1:
                cleaned_test_data.append(trans)
        else:
            cleaned_test_data.append(trans)
    train_dataset_c.data = cleaned_train_data
    test_dataset_c.data = cleaned_test_data

    out_dim = 1 + max([c[args.feature] for c in train_dataset_c.data])
    for seed in range(args.num_seeds): 
        print(f"=============== Seed: {seed} ================")
        torch.manual_seed(seed)

        cleaned_train_data = [(trans["hidden_states"].cpu(), trans["board_state"], trans[args.feature]) for trans in train_dataset_c.data]
        cleaned_test_data = [(trans["hidden_states"].cpu(), trans["board_state"], trans[args.feature]) for trans in test_dataset_c.data]
        train_dataset = ProbingDatasetCleaned(cleaned_train_data)
        test_dataset = ProbingDatasetCleaned(cleaned_test_data)
        
        for layer_name, layer_idx in layers:
    
            print(f"========= {layer_name} =========")
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True,persistent_workers=True)
            test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True,persistent_workers=True)

            probe = LinearProbe(in_dim = 7*64 if layer_name == "x" else 32*64, out_dim=out_dim, bias=False)

            probe.to(device)
            optimiser = torch.optim.AdamW(params=probe.parameters(), lr=1e-3, weight_decay=args.weight_decay)
        
            for epoch in range(1, args.num_epochs+1):
                start_time = time.time()

                precisions = [0 for _ in range(out_dim)]
                recalls = [0 for _ in range(out_dim)]
                fones = [0 for _ in range(out_dim)]
                conf_mat = [[0 for i in range(out_dim)] for j in range(out_dim)]

                for hiddens, states, targets in train_loader:
                    hiddens = states.to(torch.float).to(device) if layer_name=="x" else hiddens[:,-1,[layer_idx+c for c in channels],:,:].to(device)
                    targets = targets.to(torch.long).to(device)
                    optimiser.zero_grad()
                    logits, loss = probe(hiddens, targets)
                    loss.backward()
                    optimiser.step()
                full_acc = 0
                positive_acc = 0
                prop_pos_cor = 0
                if epoch % 1 == 0:
                    with torch.no_grad():
                        labs, preds = [], []
                        for hiddens, states, targets in test_loader:
                            hiddens = states.to(torch.float).to(device) if layer_name=="x" else hiddens[:,-1,[layer_idx+c for c in channels],:,:].to(device)
                            targets = targets.to(torch.long).to(device)
                            logits, loss = probe(hiddens, targets)
                            full_acc += (torch.sum(logits.argmax(dim=1)==targets).item())
                            preds += logits.argmax(dim=1).view(-1).tolist()
                            labs += targets.view(-1).tolist()

                        #if out_dim == 2:
                        #    prec, rec, f1, sup = precision_recall_fscore_support(labs, preds, average='binary', pos_label=1, zero_division=1, labels=[0,1])
                        #else:
                        prec, rec, f1, sup = precision_recall_fscore_support(labs, preds, average='macro', zero_division=1, labels=list(range(out_dim)))
                        
                        precisions, recalls, fones, _ = precision_recall_fscore_support(labs, preds, average=None, zero_division=1, labels=list(range(out_dim)))

                        print(f"---- Epoch {epoch} -----")
                        print("Full acc:", full_acc/(len(test_dataset.data)))
                        print("F1:", f1)
                        print("Time:", time.time()-start_time)

            results_dict = {"Acc": full_acc/(len(test_dataset.data))}
            for j in range(out_dim):
                results_dict[f"Precision_{j}"] = precisions[j]
                results_dict[f"Recall_{j}"] = recalls[j]
                results_dict[f"F1_{j}"] = fones[j]
            results_dict["Avg_F1"] = f1
            results[f"{layer_name}_hidden_states"] = results_dict

            if not os.path.exists("./results"):
                os.mkdir("./results")
            if not os.path.exists("./results/fullprobe_results"):
                os.mkdir("./results/fullprobe_results")
            if not os.path.exists("./results/fullprobe_results/models"):
                os.mkdir("./results/fullprobe_results/models")
            if not os.path.exists(f"./results/fullprobe_results/models/{args.feature}"):
                os.mkdir(f"./results/fullprobe_results/models/{args.feature}")
            torch.save(probe.state_dict(), f"./results/fullprobe_results/models/{args.feature}/{args.model_name}_{args.feature}_wd{args.weight_decay}_seed{seed}.pt")

        results_df = pd.DataFrame(results)
        results_df.to_csv(f"./results/fullprobe_results/{args.model_name}_{args.feature}_wd{args.weight_decay}_seed{seed}.csv")
