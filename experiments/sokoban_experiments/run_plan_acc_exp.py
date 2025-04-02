import torch
from create_probe_dataset import ProbingDataset, make_current_board_feature_detector, make_agent_info_extractor, make_box_info_extractor
import numpy as np
import thinker
import thinker.util as util
import gym
import gym_sokoban
import pandas as pd
import numpy as np
from thinker.actor_net import DRCNet
from train_conv_probe import ConvProbe
from sklearn.metrics import precision_recall_fscore_support
import os
import argparse



if __name__ == "__main__":
    seed = 0
    parser = argparse.ArgumentParser(description="run experiment to measure macro f1 of agent over thinking time")
    parser.add_argument("--num_layers", type=int, default=3, help="number of convlstm layers the agent has")
    parser.add_argument("--num_ticks", type=int, default=3, help="number of internal ticks the agent performs")
    parser.add_argument("--model_name", type=str, default="250m", help="name of agent checkpoint on which to run experiments")
    parser.add_argument("--num_episodes", type=int, default=1000, help="number of episodes to average over")
    parser.add_argument("--env_name", type=str, default="valid-", help="level dataset to run experiments on")
    parser.add_argument("--num_thinking_steps", type=int, default=5, help="number of thinking steps to perform prior to acting")
    args = parser.parse_args()
    num_episodes = args.num_episodes
    results = {}

    print(f"========== {args.model_name} =============")
    env = thinker.make(
        f"Sokoban-{args.env_name}v0", 
        env_n=1, 
        gpu=True if torch.cuda.is_available() else False,
        wrapper_type=1, 
        has_model=False, 
        train_model=False, 
        parallel=False, 
        save_flags=False,
        mini=True,
        mini_unqtar=False,
        mini_unqbox=False         
        ) 
    flags = util.create_setting(args=[], save_flags=False, wrapper_type=1) 
    flags.mini = True
    flags.mini_unqtar = False
    flags.mini_unqbox = False
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

    agent_loc_detector = make_current_board_feature_detector(feature_idxs=[4,5], mode="loc")
    box_loc_detector = make_current_board_feature_detector(feature_idxs=[2,3], mode="loc")
    tar_loc_detector = make_current_board_feature_detector(feature_idxs=[3,5,6], mode="loc")
    justtar_loc_detector = make_current_board_feature_detector(feature_idxs=[5,6], mode="loc")

    ckp = torch.load(ckp_path, map_location=env.device, weights_only=False)
    drc_net.load_state_dict(ckp["actor_net_state_dict"], strict=False)
    drc_net.to(env.device)

    current_board_feature_fncs = [
        ("agent_loc", agent_loc_detector),
        ("box_loc", box_loc_detector),
        ("tar_loc", tar_loc_detector),
        ("justtar_loc", justtar_loc_detector),
    ]
    future_feature_fncs = [make_box_info_extractor(), make_agent_info_extractor()]

    box_probes = []
    for layer in range(args.num_layers):
        probe = ConvProbe(32,5, 1, 0)
        probe.load_state_dict(torch.load(f"./results/convprobe_results/models/tracked_box_next_push_onto_with/{args.model_name}_layer{layer}_kernel1_wd0.001_seed0.pt", map_location=env.device, weights_only=False))
        box_probes.append(probe)


    agent_probes = []
    for layer in range(args.num_layers):
        probe = ConvProbe(32,5, 1, 0)
        probe.load_state_dict(torch.load(f"./results/convprobe_results/models/agent_onto_after/{args.model_name}_layer{layer}_kernel1_wd0.001_seed0.pt", map_location=env.device, weights_only=False))
        agent_probes.append(probe)


    for probe in box_probes + agent_probes:
        probe.to(env.device)
    rnn_state = drc_net.initial_state(batch_size=1, device=env.device)
    state = env.reset() 
    env_out = util.init_env_out(state, flags, dim_actions=1, tuple_action=False)

    episode_length = 0
    board_num = 0
    probing_data = []
    episode_entry = []

    actor_out, rnn_state = drc_net(env_out, rnn_state, greedy=True)
    trans_entry = {feature:fnc(state["real_states"][0]) for feature, fnc in current_board_feature_fncs}
    trans_entry["action"] = actor_out.action.item()
    trans_entry["value"] = round(actor_out.baseline.item(), 3) 
    trans_entry["board_state"] = state["real_states"][0].detach().cpu()
    trans_entry["hidden_states"] = drc_net.hidden_state[0].detach().cpu()
    trans_entry["board_num"] = board_num
    episode_length += 1

    while(board_num < num_episodes):

        state, reward, done, info = env.step(actor_out.action if episode_length > args.num_thinking_steps else torch.tensor([0]))
        
        if episode_length <=  (args.num_thinking_steps+1):
            for tick in list(range(1, args.num_ticks+1)):
                for layer, boxprobe in enumerate(box_probes):
                    logits, _ = boxprobe(drc_net.core.hidden_state[0,tick,(layer*64)+32:(layer*64)+64,:,:])
                    trans_entry[f"plan_box_layer{layer+1}_tick_{tick}"] = logits.argmax(dim=0).detach().cpu()
                for layer, agentprobe in enumerate(agent_probes):
                    logits, _ = agentprobe(drc_net.core.hidden_state[0,tick,(layer*64)+32:(layer*64)+64,:,:])
                    trans_entry[f"plan_agent_layer{layer+1}_tick_{tick}"] = logits.argmax(dim=0).detach().cpu()
        episode_entry.append(trans_entry)

        if done:
            for fnc in future_feature_fncs:
                episode_entry = fnc(episode_entry)
            for trans_idx, trans_entry in enumerate(episode_entry):
                trans_entry["steps_remaining"] = episode_length - trans_idx
                trans_entry["steps_taken"] = trans_idx+1
            
            probing_data += episode_entry 
            episode_length = 0
            board_num += 1
            print("Data collected from episode", board_num, "with episode length of", len(episode_entry))
            episode_entry = []
            rnn_state = drc_net.initial_state(batch_size=1, device=env.device)

        env_out = util.create_env_out(actor_out.action, state, reward, done, info, flags)
        actor_out, rnn_state = drc_net(env_out, rnn_state, greedy=True)
        trans_entry = {feature:fnc(state["real_states"][0]) for feature, fnc in current_board_feature_fncs}
        trans_entry["board_state"] = state["real_states"][0].detach().cpu()
        trans_entry["action"] = actor_out.action.item() 
        episode_length += 1

    checkpoint_results = {}
    for layer in range(args.num_layers):
        labs_a, preds_a = [[] for i in range((args.num_thinking_steps+1)*args.num_ticks)], [[] for i in range((args.num_thinking_steps+1)*args.num_ticks)]
        for trans in probing_data:
            if trans["steps_taken"] <= args.num_thinking_steps+1:
                for tick in range(args.num_ticks):
                    labs_a[(trans["steps_taken"]-1)*args.num_ticks + tick] += trans["agent_onto_after"].view(-1).tolist()
                    preds_a[(trans["steps_taken"]-1)*args.num_ticks + tick] += trans[f"plan_agent_layer{layer+1}_tick_{tick+1}"].view(-1).tolist()
        for i in range(len(labs_a)):
            prec, rec, f1, sup = precision_recall_fscore_support(labs_a[i], preds_a[i], average='macro', zero_division=1)
            checkpoint_results[f"plan_agent_layer{layer+1}_tick_{i+1}"] = f1

        labs_b, preds_b = [[] for i in range((args.num_thinking_steps+1)*args.num_ticks)], [[] for i in range((args.num_thinking_steps+1)*args.num_ticks)]
        for trans in probing_data:
            if trans["steps_taken"] <= args.num_thinking_steps+1:
                for tick in range(args.num_ticks):
                    labs_b[(trans["steps_taken"]-1)*args.num_ticks + tick] += trans["tracked_box_next_push_onto_with"].view(-1).tolist()
                    preds_b[(trans["steps_taken"]-1)*args.num_ticks + tick] += trans[f"plan_box_layer{layer+1}_tick_{tick+1}"].view(-1).tolist()
        for i in range(len(labs_b)):
            prec, rec, f1, sup = precision_recall_fscore_support(labs_b[i], preds_b[i], average='macro', zero_division=1)
            checkpoint_results[f"plan_box_layer{layer+1}_tick_{i+1}"] = f1

    results[args.model_name] = checkpoint_results
    if not os.path.exists("./results"):
        os.mkdir("./results")
    if not os.path.exists("./results/planacc_results"):
        os.mkdir("./results/planacc_results")

    pd.DataFrame({args.model_name: checkpoint_results}).to_csv(f"./results/planacc_results/{args.model_name}_planaccs_over_ticks.csv")
