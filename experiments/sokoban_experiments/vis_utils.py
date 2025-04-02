import torch
import numpy as np
import sys
import thinker
from thinker import util
import os
from thinker.actor_net import DRCNet, ResNet
from train_conv_probe import ConvProbe
from run_agent_interv_exps import agent_exp_paths, ProbeIntervDRCNet
from run_box_interv_exps import box_exp_paths
from run_cutoff_interv_exps import cutoff_exp_paths
from PIL import Image
import matplotlib.pyplot as plt

wall = Image.open("../../sokoban/gym_sokoban/envs/surface/wall_small.bmp")
wall = np.array(wall)
floor = Image.open("../../sokoban/gym_sokoban/envs/surface/floor_small.bmp")
floor = np.array(floor)
p = Image.open("../../sokoban/gym_sokoban/envs/surface/player_small.bmp")
p = np.array(p)
pot = Image.open("../../sokoban/gym_sokoban/envs/surface/player_on_target_small.bmp")
pot = np.array(pot)
b = Image.open("../../sokoban/gym_sokoban/envs/surface/box_small.bmp")
b = np.array(b)
t = Image.open("../../sokoban/gym_sokoban/envs/surface/box_target_small.bmp")
t = np.array(t)
bot = Image.open("../../sokoban/gym_sokoban/envs/surface/box_on_target_small.bmp")
bot = np.array(bot)
imgs = [wall, floor, b, bot, p, pot, t]

def start_new_traj(plan_type, level_id, model_name, num_layers, num_ticks, layer, resnet, seed=0):

    layer -= 1
    probes = {}
    probe = ConvProbe(in_channels=32,out_dim=5, kernel_size=1, padding=0)
    probe.load_state_dict(torch.load(f"./results/convprobe_results/models/tracked_box_next_push_onto_with/{model_name}_layer{layer}_kernel1_wd0.001_seed{seed}.pt", map_location=torch.device('cpu')))
    probes["box"] = probe
    probe = ConvProbe(in_channels=32,out_dim=5, kernel_size=1, padding=0)
    probe.load_state_dict(torch.load(f"./results/convprobe_results/models/agent_onto_after/{model_name}_layer{layer}_kernel1_wd0.001_seed{seed}.pt", map_location=torch.device('cpu')))
    probes["agent"] = probe

    flags = thinker.util.create_setting(args=[], save_flags=False, wrapper_type=1) 
    env = thinker.make(
                f"Sokoban-examples_clean_{level_id:04}-v0", 
                env_n=1, 
                gpu=False,
                wrapper_type=1, 
                has_model=False, 
                train_model=False, 
                parallel=False, 
                save_flags=False,
                mini=True,
                mini_unqtar=False,
                mini_unqbox=False         
            ) 
    if resnet:
        net = ResNet(
            obs_space=env.observation_space,
            action_space=env.action_space,
            flags=flags,
            record_state=True,
            num_layers=num_layers,
            hidden_dim=32
            )
    else:
        net = DRCNet(
            obs_space=env.observation_space,
            action_space=env.action_space,
            flags=flags,
            record_state=True,
            num_ticks=num_ticks,
            num_layers=num_layers
        )
    net.eval()
    ckp_path = f"../../checkpoints/sokoban/ckp_actor_realstep{model_name}.tar"
    ckp = torch.load(ckp_path, map_location=torch.device('cpu'))
    net.load_state_dict(ckp["actor_net_state_dict"], strict=False)

    rnn_state = net.initial_state(batch_size=1, device=env.device)
    state = env.reset()
    env_out = thinker.util.init_env_out(state, flags, dim_actions=1, tuple_action=False)

    traj_info = {
        "plan_type": plan_type,
        "level_id": level_id,
        "num_ticks": num_ticks,
        "layer": layer,
        "history": [],
        "resnet": resnet
    }

    return env, net, rnn_state, env_out, probes, traj_info, flags


def take_step(env, net, rnn_state, env_out, probes, flags, traj_info, pause=False, visualise=True, visualise_probs=False):

    hals = {
    40: [3, 3, 1, 0, 1],
    41: [5, 0, 1, 0, 1],
    42: [4, 6, 1, 0, 1],
    43: [4, 4, 1, 0, 1],
    44: [3, 3, 1, 0, 1],
    45: [3, 3, 0, 1, 1],
    46: [5, 0, 0, 1, 3],
    47: [4, 6, 0, 1, 2],
    48: [4, 4, 0, 1, 2],
    49: [3, 3, 0, 1, 1],
    }

    thinks = [50, 51, 52, 53, 54]

    if traj_info["plan_type"] in ["agent", "box"]:
        probe = probes[traj_info["plan_type"]]
    else:
        raise ValueError(f"{traj_info['plan_type']} is not a recognised type of plan - must be agent or box")

    if traj_info["resnet"]:
        layer_d, probe_s = 32, 0
    else:
        layer_d, probe_s = 64, 32

    actor_out, rnn_state = net(env_out, rnn_state, greedy=True)
    arrows_across_ticks = []
    for tick in range(1 if not traj_info["resnet"] else 0, traj_info["num_ticks"]+1 if not traj_info["resnet"] else 1):
        logits, _ = probe(net.hidden_state[0,tick,(traj_info["layer"]*layer_d)+probe_s:(traj_info["layer"]*layer_d)+probe_s+32,:,:])
        arrows = []
        for y in range(8):
            for x in range(8):
                action = logits.argmax(dim=0)[y,x]
                p = logits.softmax(dim=0)[action,y,x].item()
                if action == 1:
                    arrows.append((8*x+3.5,8*y+4.5, 0, 8, p)) # up
                elif action == 2:
                    arrows.append((8*x+3.5, 8*y+2, 0, -8, p)) # down
                elif action == 3:
                    arrows.append((8*x+4.5, 8*y+3.5, -8, 0, p)) # left
                elif action == 4:
                    arrows.append((8*x+2, 8*y+3.5, 8, 0, p)) # right
        arrows_across_ticks.append(arrows)


    traj_info["history"].append((env_out.real_states[0,0], arrows_across_ticks))
    
    if visualise:
        fig, axs = plt.subplots()
        img = np.zeros(shape=(64,64,3), dtype=int)
        for y in range(8):
            for x in range(8):
                img[(y*8):((y+1)*8), (x*8):((x+1)*8)] = imgs[env_out.real_states[0,0,:,y,x].argmax().item()]
        axs.imshow(img)
        axs.set_yticks([])
        axs.set_xticks([])
        for (x, y, dx, dy, p) in arrows:
            axs.arrow(x, y, dx * 0.25, dy * -0.25, head_width=4, head_length=2, color=("slateblue" if traj_info["plan_type"]=="box" else "teal"), width=2, alpha=(p if visualise_probs else 1))
    state, reward, done, info = env.step(actor_out.action if not (pause or (traj_info["level_id"] in thinks and len(traj_info["history"]) < 6)) else torch.Tensor([0]))

    if traj_info["level_id"] in hals.keys():
        y, x, old, new, n = hals[traj_info["level_id"]]
        if len(traj_info["history"]) >= n:
            state["real_states"][0, old, y, x] = 0
            state["real_states"][0, new, y, x] = 1
    env_out = thinker.util.create_env_out(actor_out.action if not (pause or (traj_info["level_id"] in thinks and len(traj_info["history"]) < 6)) else torch.Tensor([0]), state, reward, done, info, flags)

    return rnn_state, env_out


def start_new_interv_traj(interv_exp, plan_type, level_id, model_name, num_layers, num_ticks, layer, alpha, intervs, noshortroute=False, seed=0):
    flags = util.create_setting(args=[], save_flags=False, wrapper_type=1) 


    probes = {"agent": [], "box": []}
    for l in range(num_layers):
        probe = ConvProbe(32,5, 1, 0)
        probe.load_state_dict(torch.load(f"./results/convprobe_results/models/agent_onto_after/{model_name}_layer{l}_kernel1_wd0.001_seed{seed}.pt", map_location=torch.device('cpu')))
        probes["agent"].append(probe)
    for l in range(num_layers):
        probe = ConvProbe(32,5, 1, 0)
        probes["agent"].append(probe)
    for l in range(num_layers):
        probe = ConvProbe(32,5, 1, 0)
        probe.load_state_dict(torch.load(f"./results/convprobe_results/models/tracked_box_next_push_onto_with/{model_name}_layer{l}_kernel1_wd0.001_seed{seed}.pt", map_location=torch.device('cpu')))
        probes["box"].append(probe)
    for l in range(num_layers):
        probe = ConvProbe(32,5, 1, 0)
        probes["box"].append(probe)

    rot = level_id % 8
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
    else:
        raise ValueError("index problem :(")


    layer -= 1
    traj_info = {
        "plan_type": plan_type,
        "level_id": level_id,
        "num_ticks": num_ticks,
        "num_layers": num_layers,
        "layer": layer,
        "history": [],
        "up_idx": up_idx,
        "down_idx": down_idx,
        "left_idx": left_idx,
        "right_idx": right_idx,
        "patch_old": True,
        "interv_exp": interv_exp,
        "noshortroute": noshortroute
    }

    if interv_exp == "agent_shortcut":
        traj_info["exp_paths"] =  agent_exp_paths[intervs]
        traj_info["alpha_l"] = alpha * (probes["agent"][layer%num_layers].conv.weight.norm() / probes["agent"][layer].conv.weight.norm() if layer >= num_layers else 1) 
        traj_info["alpha_s"] = traj_info["alpha_l"] if not noshortroute else 0
        env = thinker.make(
                    f"Sokoban-shortcut_clean_{level_id:04}-v0", 
                    env_n=1, 
                    gpu=False,
                    wrapper_type=1, 
                    has_model=False, 
                    train_model=False, 
                    parallel=False, 
                    save_flags=False,
                    mini=True,
                    mini_unqtar=False,
                    mini_unqbox=False         
                ) 
    elif interv_exp == "box_shortcut":
        traj_info["exp_paths"] =  box_exp_paths[intervs]
        traj_info["alpha_l"] = alpha * (probes["agent"][layer%num_layers].conv.weight.norm() / probes["agent"][layer].conv.weight.norm() if layer >= num_layers else 1) 
        traj_info["alpha_s"] = traj_info["alpha_l"] if not noshortroute else 0
        env = thinker.make(
                    f"Sokoban-boxshortcut_clean_{level_id:04}-v0", 
                    env_n=1, 
                    gpu=False,
                    wrapper_type=1, 
                    has_model=False, 
                    train_model=False, 
                    parallel=False, 
                    save_flags=False,
                    mini=True,
                    mini_unqtar=False,
                    mini_unqbox=False         
                ) 
    elif interv_exp == "cutoff" or interv_exp == "cutoff_agentonly" or interv_exp == "cutoff_boxonly":
        traj_info["exp_paths"] =  cutoff_exp_paths[0]
        traj_info["alpha_a"] = alpha * (probes["agent"][layer%num_layers].conv.weight.norm() / probes["agent"][layer].conv.weight.norm() if layer >= num_layers else 1)
        traj_info["alpha_b"] = alpha * (probes["box"][layer%num_layers].conv.weight.norm() / probes["box"][layer].conv.weight.norm() if layer >= num_layers else 1)
        
        env = thinker.make(
                    f"Sokoban-cutoffpusht4_clean_{level_id:04}-v0", 
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
    else:
        raise ValueError(f"{interv_exp} is not a recognised intervention experiment")

    net = DRCNet(
                    obs_space=env.observation_space,
                    action_space=env.action_space,
                    flags=flags,
                    record_state=True,
                    num_layers=num_layers,
                    num_ticks=num_ticks
                    )
    ckp_path = "../../checkpoints/sokoban"
    ckp_path = os.path.join(util.full_path(ckp_path), f"ckp_actor_realstep{model_name}.tar")
    ckp = torch.load(ckp_path, map_location=torch.device('cpu'))
    net.load_state_dict(ckp["actor_net_state_dict"], strict=False)
    net = ProbeIntervDRCNet(net)

    rnn_state = net.drc_net.initial_state(batch_size=1, device=env.device)
    state = env.reset()
    env_out = util.init_env_out(state, flags, dim_actions=1, tuple_action=False)
    return env, net, rnn_state, env_out, probes, traj_info, flags


def take_interv_step(env, net, rnn_state, env_out, probes, flags, traj_info, pause=False, visualise=True, visualise_probs=False):
    if traj_info["interv_exp"] == "agent_shortcut":
        _, (olds, new_rs, new_ls, new_ds, new_us, checks) = traj_info["exp_paths"]
        agent_loc = (env_out.real_states[0,0,4] == 1).to(int).argmax() 
        agent_x, agent_y = agent_loc % 8, (agent_loc -(agent_loc % 8))//8
        if (agent_y, agent_x) in new_rs[traj_info["level_id"]] or (agent_y, agent_x) in new_ls[traj_info["level_id"]] or (agent_y, agent_x) in new_us[traj_info["level_id"]] or (agent_y, agent_x) in new_ds[traj_info["level_id"]]:
            traj_info["patch_old"] = False
    elif traj_info["interv_exp"] == "box_shortcut":
        _, (olds, new_rs, new_ls, new_ds, new_us, checks, boxchecks) = traj_info["exp_paths"]
        box_locs = (env_out.real_states[0,0,2] == 1).to(int).view(-1).topk(k=(env_out.real_states[0,0,2] == 1).to(int).sum()).indices.tolist()
        notonstart = 0
        for box_loc in box_locs:
            box_x, box_y = box_loc % 8, (box_loc -(box_loc % 8))//8
            if (box_y, box_x) in boxchecks[traj_info["level_id"]]:
                notonstart += 1 
        if notonstart != 1:
            print("no more patch")
            traj_info["patch_old"] = False
    elif traj_info["interv_exp"] == "cutoff" or traj_info["interv_exp"] == "cutoff_agentonly" or traj_info["interv_exp"] == "cutoff_boxonly":
        _, (olds, new_rs, new_ls, new_ds, new_us, checks, boxchecks, ags) = traj_info["exp_paths"]
        box_locs = (env_out.real_states[0,0,2] == 1).to(int).view(-1).topk(k=(env_out.real_states[0,0,2] == 1).to(int).sum()).indices.tolist()
        notonstart = 0
        for box_loc in box_locs:
            box_x, box_y = box_loc % 8, (box_loc -(box_loc % 8))//8
            if (box_y, box_x) in boxchecks[traj_info["level_id"]]:
                notonstart += 1 
        if notonstart != 1:
            traj_info["patch_old"] = False

    if traj_info["interv_exp"] == "agent_shortcut" or traj_info["interv_exp"] == "box_shortcut":
        interv_probe_type = "agent" if traj_info["interv_exp"]=="agent_shortcut" else "box"
        if traj_info["patch_old"]:
            patch_info = {traj_info["layer"] % traj_info["num_layers"]: [{"vec": probes[interv_probe_type][traj_info["layer"]].conv.weight[0].view(32), "locs": olds[traj_info["level_id"]], "alpha": traj_info["alpha_s"]},
                        {"vec": probes[interv_probe_type][traj_info["layer"]].conv.weight[traj_info["right_idx"]].view(32), "locs": new_rs[traj_info["level_id"]], "alpha": traj_info["alpha_l"]},
                        {"vec": probes[interv_probe_type][traj_info["layer"]].conv.weight[traj_info["left_idx"]].view(32), "locs": new_ls[traj_info["level_id"]], "alpha": traj_info["alpha_l"]},
                        {"vec": probes[interv_probe_type][traj_info["layer"]].conv.weight[traj_info["down_idx"]].view(32), "locs": new_ds[traj_info["level_id"]], "alpha": traj_info["alpha_l"]},
                        {"vec": probes[interv_probe_type][traj_info["layer"]].conv.weight[traj_info["up_idx"]].view(32), "locs": new_us[traj_info["level_id"]], "alpha": traj_info["alpha_l"]}] }
        else:
            patch_info = {traj_info["layer"] % traj_info["num_layers"]: [{"vec": probes[interv_probe_type][traj_info["layer"]].conv.weight[0].view(32), "locs": olds[traj_info["level_id"]], "alpha": traj_info["alpha_s"]}]}
    elif traj_info["interv_exp"] == "cutoff" or traj_info["interv_exp"] == "cutoff_agentonly" or traj_info["interv_exp"] == "cutoff_boxonly":
        if traj_info["patch_old"]:
            patch_info = {traj_info["layer"] % traj_info["num_layers"]: [
                        {"vec": probes["box"][traj_info["layer"]].conv.weight[traj_info["right_idx"]].view(32), "locs": new_rs[traj_info["level_id"]], "alpha": traj_info["alpha_b"] if traj_info["interv_exp"] != "cutoff_agentonly" else 0},
                        {"vec": probes["box"][traj_info["layer"]].conv.weight[traj_info["left_idx"]].view(32), "locs": new_ls[traj_info["level_id"]], "alpha": traj_info["alpha_b"] if traj_info["interv_exp"] != "cutoff_agentonly"  else 0},
                        {"vec": probes["box"][traj_info["layer"]].conv.weight[traj_info["down_idx"]].view(32), "locs": new_ds[traj_info["level_id"]], "alpha": traj_info["alpha_b"] if traj_info["interv_exp"] != "cutoff_agentonly"  else 0},
                        {"vec": probes["box"][traj_info["layer"]].conv.weight[traj_info["up_idx"]].view(32), "locs": new_us[traj_info["level_id"]], "alpha": traj_info["alpha_b"] if traj_info["interv_exp"] != "cutoff_agentonly"  else 0},
                        {"vec": probes["agent"][traj_info["layer"]].conv.weight[traj_info["up_idx"]].view(32), "locs": ags[traj_info["level_id"]], "alpha": traj_info["alpha_a"] if traj_info["interv_exp"] != "cutoff_boxonly"  else 0}
                        ]}
            print(traj_info["patch_old"])
        else:
            patch_info = {traj_info["layer"] % traj_info["num_layers"]: [{"vec": probes["box"][traj_info["layer"]].conv.weight[0].view(32), "locs": [], "alpha": 0}]}

    patch_action, patch_action_probs, patch_logits, rnn_state, value = net.forward_patch(env_out, rnn_state, activ_ticks=list(range(traj_info["num_ticks"])),
                                                            patch_info=patch_info)

    
    arrows_across_ticks = []
    for tick in range(1, traj_info["num_ticks"]+1):
        logits, _ = probes[traj_info["plan_type"]][traj_info["layer"]](net.drc_net.core.hidden_state[0,tick,(64*traj_info["layer"])+32:(64*traj_info["layer"])+64,:,:])
        arrows = []
        for y in range(8):
            for x in range(8):
                action = logits.argmax(dim=0)[y,x]
                p = logits.softmax(dim=0)[action,y,x].item()
                if action == 1:
                    arrows.append((8*x+3.5,8*y+4.5, 0, 8, p)) # up
                elif action == 2:
                    arrows.append((8*x+3.5, 8*y+2, 0, -8, p)) # down
                elif action == 3:
                    arrows.append((8*x+4.5, 8*y+3.5, -8, 0, p)) # left
                elif action == 4:
                    arrows.append((8*x+2, 8*y+3.5, 8, 0, p)) # right
        arrows_across_ticks.append(arrows)

    traj_info["history"].append((env_out.real_states[0,0], arrows_across_ticks))

    if visualise:
        fig, axs = plt.subplots()
        img = np.zeros(shape=(64,64,3), dtype=int)
        for y in range(8):
            for x in range(8):
                img[(y*8):((y+1)*8), (x*8):((x+1)*8)] = imgs[env_out.real_states[0,0,:,y,x].argmax().item()]
        axs.imshow(img)
        axs.set_yticks([])
        axs.set_xticks([])
        for (x, y, dx, dy, p) in arrows:
            axs.arrow(x, y, dx * 0.25, dy * -0.25, head_width=4, head_length=2, color=("slateblue" if traj_info["plan_type"]=="box" else "teal"), width=2, alpha=(p if visualise_probs else 1))
    state, reward, done, info = env.step(patch_action)
    env_out = util.create_env_out(patch_action, state, reward, done, info, flags)
    return rnn_state, env_out


def visualise_plan(traj_info, include_captions=True, visualise_all_ticks=False, visualise_probs=False, visualise_only_final_trans_final_tick=False, save_img=False, save_img_name="test"):
    
    if visualise_only_final_trans_final_tick:
        fig, axs = plt.subplots(1, 1, figsize=(4 , 4)
            )
        state, arrows_across_ticks = traj_info["history"][-1]
        img = np.zeros(shape=(64,64,3), dtype=int)
        for y in range(8):
            for x in range(8):
                img[(y*8):((y+1)*8), (x*8):((x+1)*8)] = imgs[state[:,y,x].argmax().item()]

        axs.imshow(img)
        axs.set_yticks([])
        axs.set_xticks([])
        for (x, y, dx, dy, p) in arrows_across_ticks[-1]:
            axs.arrow(x, y, dx * 0.25, dy * -0.25, head_width=4, head_length=2, color=("slateblue" if traj_info["plan_type"]=="box" else "teal"), width=2, alpha=(p if visualise_probs else 1))
        
        if include_captions:
            axs.set_xlabel(f"step {traj_info['history']}, tick {traj_info['num_ticks']}", fontsize=20)
    else:
        fig, axs = plt.subplots(
            1,
            len(traj_info["history"]) * (traj_info["num_ticks"] if visualise_all_ticks else 1),
            figsize=(len(traj_info["history"]) * 4 * (traj_info["num_ticks"] if visualise_all_ticks else 1), 4)
            )


        for i, (state, arrows_across_ticks) in enumerate(traj_info["history"]):
            img = np.zeros(shape=(64,64,3), dtype=int)
            for y in range(8):
                for x in range(8):
                    img[(y*8):((y+1)*8), (x*8):((x+1)*8)] = imgs[state[:,y,x].argmax().item()]
            if visualise_all_ticks:
                for k in range(traj_info["num_ticks"]):
                    axs[i*traj_info["num_ticks"]+k].imshow(img)
                    axs[i*traj_info["num_ticks"]+k].set_yticks([])
                    axs[i*traj_info["num_ticks"]+k].set_xticks([])
                    for (x, y, dx, dy, p) in arrows_across_ticks[k]:
                        axs[i*traj_info["num_ticks"]+k].arrow(x, y, dx * 0.25, dy * -0.25, head_width=4, head_length=2, color=("slateblue" if traj_info["plan_type"]=="box" else "teal"), width=2, alpha=(p if visualise_probs else 1))
                    if include_captions:
                        axs[i*traj_info["num_ticks"]+k].set_xlabel(f"step {i}, tick {k+1}", fontsize=20)
            else:
                axs[i].imshow(img)
                axs[i].set_yticks([])
                axs[i].set_xticks([])
                for (x, y, dx, dy, p) in arrows_across_ticks[-1]:
                    axs[i].arrow(x, y, dx * 0.25, dy * -0.25, head_width=4, head_length=2, color=("slateblue" if traj_info["plan_type"]=="box" else "teal"), width=2, alpha=(p if visualise_probs else 1))
                if include_captions:
                    axs[i].set_xlabel(f"step {i}, tick {traj_info['num_ticks']}", fontsize=20)
    
    if save_img:
        if not os.path.exists("./plan_visualisations"):
            os.mkdir("./plan_visualisations")
        plt.savefig(f"./plan_visualisations/{save_img_name}.png", format="png", bbox_inches="tight")
