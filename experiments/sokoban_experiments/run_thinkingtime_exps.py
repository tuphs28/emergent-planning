import torch
import numpy as np
import thinker
import thinker.util as util
import gym
import gym_sokoban
import pandas as pd
import numpy as np
from thinker.actor_net import DRCNet
import os
import pandas as pd
import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="run thinking time experiments to measure extent to which agent solves extra levels when given thinking time")
    parser.add_argument("--num_episodes", type=int, default=1000, help="number of episodes to run experiment for")
    parser.add_argument("--env_name", type=str, default="med", help="level dataset to run experiments on")
    parser.add_argument("--num_thinking_steps", type=int, default=5, help="number of extra thinking steps to perform")
    parser.add_argument("--model_name", type=str, default="250m", help="name of agent checkpoint on which to run experiments")
    parser.add_argument("--num_layers", type=int, default=3, help="number of convlstm layers the agent has")
    parser.add_argument("--num_ticks", type=int, default=3, help="number of internal ticks the agent performs")
    parser.add_argument('--range', action='store_true', help="whether to run the experiment for checkpoints 1m50m")
    args = parser.parse_args()

    env_n = 1
    flags = util.create_setting(args=[], save_flags=False, wrapper_type=1)
    flags.mini = True
    flags.mini_unqtar = False
    flags.mini_unqbox = False
    mini_sokoban = True
    results = []
    for num_thinking_steps in [0, args.num_thinking_steps]:
        print(f"==== ********** STEPS: {num_thinking_steps} ********** ====")

        if args.range:
            agents = [f"{k}m" for k in list(range(1,51))]
        else:
            agents = [args.model_name]

        for agent_name in agents: 
            print(f"==== Running agent {agent_name} ====")
            ep_num = 0
            solve_count = 0
            all_count = 0

            for j in range(args.num_episodes):
                env = thinker.make(
                f"Sokoban-{args.env_name}_clean_{j:04}-v0", 
                    env_n=env_n, 
                    gpu= True if torch.cuda.is_available() else False,
                    wrapper_type=1, 
                    has_model=False,
                    train_model=False, 
                    parallel=False, 
                    save_flags=False,
                    mini=mini_sokoban, 
                    mini_unqtar = False,
                    mini_unqbox=False    
                ) 

                if j == 0:
                    flags = util.create_setting(args=[], save_flags=False, wrapper_type=1) 
                    flags.mini = True
                    flags.mini_unqtar = False
                    flags.mini_unqbox = False
                    drc_net = DRCNet(
                        obs_space=env.observation_space,
                        action_space=env.action_space,
                        flags=flags,
                        record_state=False,
                        num_ticks=args.num_ticks,
                        num_layers=args.num_layers
                        )
                    drc_net.to(env.device)
                    ckp = torch.load(f"../../checkpoints/sokoban/ckp_actor_realstep{agent_name}.tar", map_location=env.device, weights_only=False)
                    drc_net.load_state_dict(ckp["actor_net_state_dict"], strict=False)
                state = env.reset() 

                step_count = 0
            
                states = []
                rnn_state = drc_net.initial_state(batch_size=env_n, device=env.device)
                env_out = util.init_env_out(state, flags, dim_actions=1, tuple_action=False)
                for _ in range(num_thinking_steps):
                    actor_out, rnn_state = drc_net(env_out, rnn_state, greedy=True)
                    state, reward, done, info = env.step(torch.tensor([0]))
                    env_out = util.create_env_out(torch.tensor([0]), state, reward, done, info, flags)
                actor_out, rnn_state = drc_net(env_out, rnn_state, greedy=True)
                state, reward, done, info = env.step(actor_out.action)
                step_count += 1

                while not done:
                    states.append(state["real_states"][0])
                    env_out = util.create_env_out(actor_out.action, state, reward, done, info, flags)
                    with torch.no_grad():
                        actor_out, rnn_state = drc_net(env_out, rnn_state, greedy=True)

                    state, reward, done, info = env.step(actor_out.action)
                    step_count += 1

                if step_count < 115-num_thinking_steps:
                    solve_count += 1
            results.append({"success_rate": solve_count / args.num_episodes, "agent": agent_name, "thinking_steps": num_thinking_steps})
    
    if not os.path.exists("./results"):
        os.mkdir("./results")
    if not os.path.exists("./results/thinkingtime_results"):
        os.mkdir("./results/thinkingtime_results")
    pd.DataFrame(results).to_csv(f"./results/thinkingtime_results/" + (f"{args.model_name}" if not args.range else "1m-50m") +f"_{args.env_name}.csv")
