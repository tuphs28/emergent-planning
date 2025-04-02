import torch
import torch.nn as nn
import thinker
import thinker.util as util
import gym
import gym_sokoban
import pandas as pd
import numpy as np
from thinker.actor_net import DRCNet
from train_conv_probe import ConvProbe
import os
from thinker.actor_net import sample
from thinker.util import EnvOut
from run_agent_interv_exps import ProbeIntervDRCNet
import argparse

box_intervs = [
    ([], [(4,3)], [], [], [], [], [(4,3)], [(3,3)]),
    ([], [(3,4)], [], [], [], [], [(3,4)], [(2,4)]),
    ([], [(2,1)], [], [], [], [], [(2,1)], [(1,1)]),
    ([], [], [(4,3)], [], [], [], [(4,3)], [(3,3)]),
    ([], [(4,1)], [], [], [], [], [(4,1)], [(3,1)]),
    ([], [], [(4,3)], [], [], [], [(4,3)], [(3,3)]),
    ([], [(4,1)], [], [], [], [], [(4,1)], [(3,1)]),
    ([], [(2,2)], [], [], [], [], [(2,2)], [(1,2)]),
    ([], [], [(2,5)], [], [], [], [(2,5)], [(1,5)]),
    ([], [(2,5)], [], [], [], [], [(2,5)], [(1,5)]),
    ([], [(2,2)], [], [], [], [], [(2,2)], [(1,2)]),
    ([], [], [(3,4)], [], [], [], [(3,4)], [(2,4)]),
    ([], [(4,3)], [], [], [], [], [(4,3)], [(3,3)]),
    ([], [], [(4,3)], [], [], [], [(4,3)], [(3,3)]),
    ([], [], [(4,3)], [], [], [], [(4,3)], [(3,3)]),
    ([], [(2,2)], [], [], [], [], [(2,2)], [(1,2)]),
    ([], [(2,2)], [], [], [], [], [(2,2)], [(1,2)]),
    ([], [(2,2)], [], [], [], [], [(2,2)], [(1,2)]),
    ([], [], [(2,2)], [], [], [], [(2,2)], [(1,2)]), 
    ([], [], [(4,3)], [], [], [], [(4,3)], [(3,3)]),
    ([], [], [(4,3)], [], [], [], [(4,3)], [(3,3)]), 
    ([], [(2,2)], [], [], [], [], [(2,2)], [(1,2)]),
    ([], [], [(2,2)], [], [], [], [(2,2)], [(1,2)]), 
    ([], [], [(2,5)], [], [], [], [(2,5)], [(1,5)]), 
    ([], [(2,5)], [], [], [], [], [(2,5)], [(1,5)]), 
]


cutoff_exp_paths = []
for name, paths in zip([1], [box_intervs]):
    olds, new_rs, new_ls, new_ds, new_us, checks, boxchecks, ags = [], [], [], [], [], [], [], []
    for old_path, new_right, new_left, new_down, new_up, checkpoint, boxpoint, ag in paths:
        old_path_1 = [(x,7-y) for y,x in old_path]
        new_right_1 = [(x,7-y) for y,x in new_right]
        new_left_1 = [(x,7-y) for y,x in new_left]
        new_down_1 = [(x,7-y) for y,x in new_down]
        new_up_1 = [(x,7-y) for y,x in new_up]
        checkpoint_1 = [(x,7-y) for y,x in checkpoint]
        boxpoint_1 = [(x,7-y) for y,x in boxpoint]
        ag_1 = [(x,7-y) for y,x in ag]

        old_path_2 = [(7-y,7-x) for y,x in old_path]
        new_right_2 = [(7-y,7-x) for y,x in new_right]
        new_left_2 = [(7-y,7-x) for y,x in new_left]
        new_down_2 = [(7-y,7-x) for y,x in new_down]
        new_up_2 = [(7-y,7-x) for y,x in new_up]
        checkpoint_2 = [(7-y,7-x) for y,x in checkpoint]
        boxpoint_2 = [(7-y,7-x) for y,x in boxpoint]
        ag_2 = [(7-y,7-x) for y,x in ag]

        old_path_3 = [(7-x,y) for y,x in old_path]
        new_right_3 = [(7-x,y) for y,x in new_right]
        new_left_3 = [(7-x,y) for y,x in new_left]
        new_down_3 = [(7-x,y) for y,x in new_down]
        new_up_3 = [(7-x,y) for y,x in new_up]
        checkpoint_3 = [(7-x,y) for y,x in checkpoint]
        boxpoint_3 = [(7-x,y) for y,x in boxpoint]
        ag_3 = [(7-x,y) for y,x in ag]

        old_path_4 = [(y,7-x) for y,x in old_path]
        new_right_4 = [(y,7-x) for y,x in new_right]
        new_left_4 = [(y,7-x) for y,x in new_left]
        new_down_4 = [(y,7-x) for y,x in new_down]
        new_up_4 = [(y,7-x) for y,x in new_up]
        checkpoint_4 = [(y,7-x) for y,x in checkpoint]
        boxpoint_4 = [(y,7-x) for y,x in boxpoint]
        ag_4 = [(y,7-x) for y,x in ag]

        old_path_5 = [(x,7-y) for y,x in old_path_4]
        new_right_5 = [(x,7-y) for y,x in new_right_4]
        new_left_5 = [(x,7-y) for y,x in new_left_4]
        new_down_5 = [(x,7-y) for y,x in new_down_4]
        new_up_5 = [(x,7-y) for y,x in new_up_4]
        checkpoint_5 = [(x,7-y) for y,x in checkpoint_4]
        boxpoint_5 = [(x,7-y) for y,x in boxpoint_4]
        ag_5 = [(x,7-y) for y,x in ag_4]

        old_path_6 = [(7-y,7-x) for y,x in old_path_4]
        new_right_6 = [(7-y,7-x) for y,x in new_right_4]
        new_left_6 = [(7-y,7-x) for y,x in new_left_4]
        new_down_6 = [(7-y,7-x) for y,x in new_down_4]
        new_up_6 = [(7-y,7-x) for y,x in new_up_4]
        checkpoint_6 = [(7-y,7-x)  for y,x in checkpoint_4]
        boxpoint_6 = [(7-y,7-x)  for y,x in boxpoint_4]
        ag_6 = [(7-y,7-x)  for y,x in ag_4]

        old_path_7 = [(7-x,y) for y,x in old_path_4]
        new_right_7 = [(7-x,y) for y,x in new_right_4]
        new_left_7 = [(7-x,y) for y,x in new_left_4]
        new_down_7 = [(7-x,y) for y,x in new_down_4]
        new_up_7 = [(7-x,y) for y,x in new_up_4]
        checkpoint_7 = [(7-x,y) for y,x in checkpoint_4]
        boxpoint_7 = [(7-x,y) for y,x in boxpoint_4]
        ag_7 = [(7-x,y) for y,x in ag_4]

        olds += [old_path, old_path_1, old_path_2, old_path_3, old_path_4, old_path_5, old_path_6, old_path_7]
        new_rs += [new_right, new_right_1, new_right_2, new_right_3, new_right_4, new_right_5, new_right_6, new_right_7]
        new_ls += [new_left, new_left_1, new_left_2, new_left_3, new_left_4, new_left_5, new_left_6, new_left_7]
        new_ds += [new_down, new_down_1, new_down_2, new_down_3, new_down_4, new_down_5, new_down_6, new_down_7]
        new_us += [new_up, new_up_1, new_up_2, new_up_3, new_up_4, new_up_5, new_up_6, new_up_7]
        checks += [checkpoint, checkpoint_1, checkpoint_2, checkpoint_3, checkpoint_4, checkpoint_5, checkpoint_6, checkpoint_7]
        boxchecks += [boxpoint, boxpoint_1, boxpoint_2, boxpoint_3, boxpoint_4, boxpoint_5, boxpoint_6, boxpoint_7]
        ags += [ag, ag_1, ag_2, ag_3, ag_4, ag_5, ag_6, ag_7]
    cutoff_exp_paths.append([name, (olds, new_rs, new_ls, new_ds, new_us, checks, boxchecks, ags)])

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="run intervention experiments in Cutoff levels")
    parser.add_argument("--model_name", type=str, default="250m", help="name of agent checkpoint on which to run experiments")
    parser.add_argument("--num_seeds", type=int, default=5, help="number of probes to run experiments for")
    parser.add_argument("--num_layers", type=int, default=3, help="number of convlstm layers the agent has")
    parser.add_argument("--num_ticks", type=int, default=3, help="number of internal ticks the agent performs (DRC only)")
    parser.add_argument("--num_episodes", type=int, default=200, help="number of Cutoff episodes to intervene in")
    parser.add_argument("--noagentinterv", action="store_true", help="flag to not perform agent intervention")
    parser.add_argument("--noboxinterv", action="store_true", help="flag to not perform agent intervention")
    args = parser.parse_args()

    device = torch.device("cpu")
    flags = util.create_setting(args=[], save_flags=False, wrapper_type=1) 
    for seed in list(range(args.num_seeds)):

        box_probes = []
        agent_probes = []
        for l in range(args.num_layers):
            probe = ConvProbe(32,5, 1, 0)
            probe.load_state_dict(torch.load(f"./results/convprobe_results/models/tracked_box_next_push_onto_with/{args.model_name}_layer{l}_kernel1_wd0.001_seed{seed}.pt", map_location=torch.device('cpu'), weights_only=False))
            box_probes.append(probe)
        for l in range(args.num_layers):
            torch.manual_seed((seed*args.num_layers)+l)
            probe = ConvProbe(32,5, 1, 0)
            box_probes.append(probe)
        for l in range(args.num_layers):
            probe = ConvProbe(32,5, 1, 0)
            probe.load_state_dict(torch.load(f"./results/convprobe_results/models/agent_onto_after/{args.model_name}_layer{l}_kernel1_wd0.001_seed{seed}.pt", map_location=torch.device('cpu'), weights_only=False))
            agent_probes.append(probe)
        for l in range(args.num_layers):
            torch.manual_seed((seed*args.num_layers)+l)
            probe = ConvProbe(32,5, 1, 0)
            agent_probes.append(probe)
        for probe in agent_probes + box_probes:
            probe.to(device)
        
        results = []
        for layer in range(args.num_layers*2):
            for interv, (olds, new_rs, new_ls, new_ds, new_us, checks, boxchecks, ags) in cutoff_exp_paths:
                for alpha in [0.25,0.5,1,2,4]:
                    alpha_t = alpha
                    if layer >= args.num_layers:
                        alpha_a = alpha * agent_probes[layer%args.num_layers].conv.weight.norm() / agent_probes[layer].conv.weight.norm()
                        alpha_b = alpha * box_probes[layer%args.num_layers].conv.weight.norm() / box_probes[layer].conv.weight.norm()
                    else:
                        alpha_a = alpha
                        alpha_b = alpha
                    print(f"========================================= {layer=}, {alpha=}, {interv=}, {seed=}==================================")
                    successes = 0
                    alls = []
                    for j in range(0,args.num_episodes):
                        steps = []
                        env = thinker.make(
                                    f"Sokoban-cutoffpusht4_clean_{j:04}-v0", 
                                    env_n=1, 
                                    gpu= True if torch.cuda.is_available() else False,
                                    wrapper_type=1, 
                                    has_model=False, 
                                    train_model=False, 
                                    parallel=False, 
                                    save_flags=False,
                                    mini=True,
                                    mini_unqtar=False,
                                    mini_unqbox=False         
                                ) 
                        if j == 0 and layer == 0 and seed == 0:
                            drc_net = DRCNet(
                                            obs_space=env.observation_space,
                                            action_space=env.action_space,
                                            flags=flags,
                                            record_state=True,
                                            num_layers=args.num_layers,
                                            num_ticks=args.num_ticks
                                            )
                            ckp_path = "../../checkpoints/sokoban"
                            ckp_path = os.path.join(util.full_path(ckp_path), f"ckp_actor_realstep{args.model_name}.tar")
                            ckp = torch.load(ckp_path, map_location=torch.device('cpu'), weights_only=False)
                            drc_net.load_state_dict(ckp["actor_net_state_dict"], strict=False)
                            drc_net.to(device)
                            patch_net = ProbeIntervDRCNet(drc_net)

                        rnn_state = drc_net.initial_state(batch_size=1, device=env.device)
                        state = env.reset()
                        env_out = util.init_env_out(state, flags, dim_actions=1, tuple_action=False)

                        patch_old = True
                        done = False
                        ep_len = 0

                        rot = j % 8
                        if rot in [3,5]:
                            right_idx = 1
                            left_idx = 2
                        elif rot in [1,7]:
                            right_idx = 2
                            left_idx = 1
                        elif rot in [2,4]:
                            right_idx = 3
                            left_idx = 4
                        elif rot in [0,6]:
                            right_idx = 4
                            left_idx = 3
                        else:
                            raise ValueError("index problem :(")

                        if rot in [0,4]:
                            down_idx = 2
                            up_idx = 1
                        elif rot in [1,5]:
                            down_idx = 3
                            up_idx = 4
                        elif rot in [2,6]:
                            down_idx = 1
                            up_idx = 2
                        elif rot in [3,7]:
                            down_idx = 4
                            up_idx = 3

                        while not done:
                            box_locs = (state["real_states"][0][2] == 1).to(int).view(-1).topk(k=(state["real_states"][0][2] == 1).to(int).sum()).indices.tolist()
                            notonstart = 0
                            for box_loc in box_locs:
                                box_x, box_y = box_loc % 8, (box_loc -(box_loc % 8))//8
                                if (box_y, box_x) in boxchecks[j]:
                                    notonstart += 1 
                            if notonstart != 1:
                                patch_old = False

                            if patch_old:
                                patch_info = {layer % args.num_layers: [
                                            {"vec": box_probes[layer].conv.weight[right_idx].view(32), "locs": new_rs[j], "alpha": alpha_b if not args.noboxinterv else 0},
                                            {"vec": box_probes[layer].conv.weight[left_idx].view(32), "locs": new_ls[j], "alpha": alpha_b if not args.noboxinterv else 0},
                                            {"vec": box_probes[layer].conv.weight[down_idx].view(32), "locs": new_ds[j], "alpha": alpha_b if not args.noboxinterv else 0},
                                            {"vec": box_probes[layer].conv.weight[up_idx].view(32), "locs": new_us[j], "alpha": alpha_b if not args.noboxinterv else 0},
                                            {"vec": agent_probes[layer].conv.weight[up_idx].view(32), "locs": ags[j], "alpha": alpha_a if not args.noagentinterv else 0}
                                            ]}
                            else:
                                patch_info = {layer % args.num_layers: [{"vec": box_probes[layer].conv.weight[0].view(32), "locs": [], "alpha": 0}]}
                            patch_action, patch_action_probs, patch_logits, rnn_state, value = patch_net.forward_patch(env_out, rnn_state, activ_ticks=list(range(args.num_ticks)),
                                                                                    patch_info=patch_info)
                            state, reward, done, info = env.step(patch_action)
                            ep_len += 1
                            env_out = util.create_env_out(patch_action, state, reward, done, info, flags)

                        if ep_len < 115:
                            successes += 1
                        else:
                            pass
                    print(successes / args.num_episodes)
                    results.append({"layer": layer, "alpha": alpha_t, "success_rate": (successes / args.num_episodes), "intervs": interv})
        
            
        if not os.path.exists("./results"):
            os.mkdir("./results")
        if not os.path.exists("./results/interv_results"):
            os.mkdir("./results/interv_results")
        pd.DataFrame(results).to_csv(f"interv_results/cutoffinterv" + ("" if args.noboxinterv else "_boxinterv") + ("" if args.noagentinterv else "_agentinterv") + f"_{args.model_name}_seed{seed}test.csv")