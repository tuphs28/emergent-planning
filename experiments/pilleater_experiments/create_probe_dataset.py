import sys
from thinker.main import make, Env
import thinker
from thinker.actor_net import DRCNet, ResNet
from torch.utils.data.dataset import Dataset
import torch
from torch.nn.functional import relu
from thinker import util
from typing import Callable, NamedTuple, Optional
from numpy.random import uniform
import os
import argparse
import gym

def make_current_board_feature_detector(feature_idxs: list, mode: str) -> Callable:
    """Create feature detector functions to extract discrete features from mini-sokoban boards. Boards must be (7,8,8) arrays

    Args:
        feature_idxs (list): list index of feature of interest (see sokoban.cpp);
        mode (str): type of feature detector to construct: "adj" (to count number of adjacent features), "num" (to count total number of features on board) or "loc" (extract location of features)

    Returns:
        Callable: feature detector function, takes in a board state and returns the desired feature
    """
    if mode == "num":
        def feature_detector(board: torch.tensor) -> int:
            return sum([torch.sum((board[feature_idx,:,:]==1).int()) for feature_idx in feature_idxs]).item()
    elif mode == "loc":
        def feature_detector(board):
            locs_xy = sum([(board[feature_idx,:,:]==1) for feature_idx in feature_idxs]).nonzero()
            locs = tuple([(13*x+y).item() for (x,y) in locs_xy]) # each location is an int in range [0,63]
            return locs
    else:
        raise ValueError(f"Please enter a valid mode to construct a feature detector - user entered {mode}, valid modes are adj, num and loc")
    return feature_detector

def make_future_feature_detector(feature_name: str, mode: str, steps_ahead: Optional[int] = None) -> Callable:
    """Create function that adds a feature to each transition (i.e. a dictionary of features) corresponding to the feature with name feature_name in steps_ahead steps

    Args:
        feature_name (str): feature to track steps_ahead into the future
        steps_ahead (Optional int): number of steps ahead into the future to look for this feature if mode is either ahead or traj
        mode (str): type of feature detector to construct: ahead (make feature corresponding to feature_name in steps_ahead steps), traj (make feature corresponding to trajectory of feature_name from current value to over steps_ahead steps) or change (number of steps until the feature next changes)

    Returns:
        Callable: feature detector function, takes in a list of transitions for a single episode and adds an entry for feature_name in steps_ahead steps
    """
    if mode == "ahead":
        new_feature_name = f"{feature_name}_ahead_{steps_ahead}"
        def feature_detector(episode_entry: list) -> list:
            assert feature_name in episode_entry[0].keys(), f"Error: This feature detector has been set up to track {feature_name} which is not contained in the episode entry - please re-create it using one of the following features: {episode_entry[0].keys()}"
            episode_length = len(episode_entry)
            for trans_idx, trans_entry in enumerate(episode_entry):
                trans_entry[new_feature_name] = episode_entry[trans_idx+steps_ahead][feature_name] if trans_idx < episode_length-steps_ahead-1 else -1
            return episode_entry
    elif mode == "traj":
        new_feature_name = f"{feature_name}_traj_{steps_ahead}"
        def feature_detector(episode_entry: list) -> list:
            assert feature_name in episode_entry[0].keys(), f"Error: This feature detector has been set up to track {feature_name} which is not contained in the episode entry - please re-create it using one of the following features: {episode_entry[0].keys()}"
            episode_length = len(episode_entry)
            for trans_idx, trans_entry in enumerate(episode_entry):
                traj = []
                if trans_idx < episode_length-steps_ahead-1:
                    for traj_idx in range(steps_ahead+1):
                        traj.append(episode_entry[trans_idx+traj_idx][feature_name])
                    trans_entry[new_feature_name] = tuple(traj)
                else:
                    trans_entry[new_feature_name] = -1
            return episode_entry
    else:
        raise ValueError(f"User entered mode {mode}, valid modes are: ahead, traj, change")
    return feature_detector

def make_trajectory_detector(steps_ahead: int, feature_name: str, inc_current: bool = False) -> Callable:

    def get_future_trajectories(episode_entry: list) -> list:
        virtual_ext = [{feature_name: episode_entry[-1][feature_name]}]
        for trans_idx, trans in enumerate(episode_entry):
            feature_locs_xy = []
            for future_trans in (episode_entry+virtual_ext)[trans_idx+(1 if not inc_current else 0):trans_idx+steps_ahead+(2 if not inc_current else 1)]:
                feature_locs_xy += [(future_trans[feature_name][feature_idx] % 13, (future_trans[feature_name][feature_idx]-(future_trans[feature_name][feature_idx]%13))//13) for feature_idx in range(len(future_trans[feature_name]))]
            feature_locs_xy = torch.tensor(feature_locs_xy)
            trajectory = torch.zeros(size=(13,13), dtype=torch.long)
            if len(feature_locs_xy.shape) != 1:
                trajectory[feature_locs_xy[:,1],feature_locs_xy[:,0]] = 1
            trans[f"{feature_name}_{'future_trajectory' if steps_ahead!=0 else 'current'}_{steps_ahead}"] = trajectory
        return episode_entry

    return get_future_trajectories


def generate_aug_trans(episode_entry):
    trans = episode_entry[-1]
    agent_loc = trans["agent_loc"][0]
    agent_loc = ((agent_loc -(agent_loc % 13))//13, agent_loc % 13,)
    agent_y, agent_x = agent_loc
    wall_locs = [((wall_loc-(wall_loc % 13))//13, wall_loc % 13) for wall_loc in trans["board_state"][0].view(-1).topk(k=(trans["board_state"][0]==1).to(int).sum()).indices]
    if trans["action"] == 1:
        if agent_y > 1 and (agent_y-1,agent_x) not in wall_locs:
            new_agent_loc = (agent_y-1,agent_x)
        else:
            new_agent_loc = agent_loc
    elif trans["action"] == 2:
        if agent_y < 12 and (agent_y+1,agent_x) not in wall_locs:
            new_agent_loc = (agent_y+1,agent_x)
        else:
            new_agent_loc = agent_loc
    elif trans["action"] == 3:
        if agent_x > 1  and (agent_y,agent_x-1) not in wall_locs:
            new_agent_loc = (agent_y,agent_x-1)
        else:
            new_agent_loc = agent_loc
    elif trans["action"] == 4:
        if agent_x < 12 and (agent_y,agent_x+1) not in wall_locs:
           new_agent_loc = (agent_y,agent_x+1)
        else:
            new_agent_loc = agent_loc
    else:
            new_agent_loc = agent_loc
    new_agent_loc = tuple([13*new_agent_loc[0] + new_agent_loc[1]])
    trans = { "agent_loc": new_agent_loc, "action": 0}
    return trans

def make_agent_info_extractor(ahead = 100) -> Callable:
    def agent_info_extractor(episode_entry: list) -> list:
        # track squares from which agent performs actions to leave
        aug_episode_entry = episode_entry + [generate_aug_trans(episode_entry)]
        for trans_idx, trans in enumerate(aug_episode_entry):
            board_locs = torch.zeros((13,13), dtype=int)
            #cur_level = trans["level"]
            for loc_idx in range(169):
                for future_trans_idx, future_trans in enumerate(aug_episode_entry[trans_idx:-1][:ahead]):
                    if loc_idx in future_trans["agent_loc"] and (future_trans["agent_loc"] != aug_episode_entry[trans_idx+future_trans_idx+1]["agent_loc"]): #NB: ignore no-ops and effective no-ops since want action we leave square with 
                        board_locs[(loc_idx-loc_idx%13)//13, loc_idx%13] = future_trans["action"]
                        break
                    #elif future_trans["level"] != cur_level:
                        #break
            trans[f"agent_onto_with_{ahead}"] = board_locs
            new_board_locs = torch.zeros((13,13), dtype=int)
            new_board_locs[board_locs != 0 ] = 1
            trans[f"agent_from_{ahead}"] = new_board_locs
        episode_entry = aug_episode_entry[:-1]
        # track squares from which agent performs action to enter
        aug_episode_entry = episode_entry + [generate_aug_trans(episode_entry)]
        for trans_idx, trans in enumerate(aug_episode_entry):
            board_locs = torch.zeros((13,13), dtype=int)
            #cur_level = trans["level"]
            for loc_idx in range(169):
                for future_trans_idx, future_trans in enumerate(aug_episode_entry[trans_idx+1:][:ahead]):
                    if loc_idx in future_trans["agent_loc"] and aug_episode_entry[trans_idx+future_trans_idx]["agent_loc"] != future_trans["agent_loc"]:
                        board_locs[(loc_idx-loc_idx%13)//13, loc_idx%13] = aug_episode_entry[trans_idx+future_trans_idx]["action"]
                        break
                    #elif future_trans["level"] != cur_level:
                        #break
            trans[f"agent_onto_after_{ahead}"] = board_locs
            new_board_locs = torch.zeros((13,13), dtype=int)
            new_board_locs[board_locs != 0 ] = 1
            trans[f"agent_onto_{ahead}"] = new_board_locs
        episode_entry = aug_episode_entry[:-1]
        for trans_idx, trans in enumerate(episode_entry):
            board_locs = torch.zeros((13,13), dtype=int)
            #cur_level = trans["level"]
            for loc_idx in range(169):
                for future_trans in episode_entry[trans_idx+1:][:ahead]:
                    if loc_idx in future_trans["agent_loc"]:
                        board_locs[(loc_idx-loc_idx%13)//13, loc_idx%13] += (1 if board_locs[(loc_idx-loc_idx%13)//13, loc_idx%13] <= 2 else 0)
                    #elif future_trans["level"] != cur_level:
                        #break
            trans["agent_loc_count"] = board_locs
        return episode_entry
    return agent_info_extractor

@torch.no_grad()
def create_probing_data(net: DRCNet, env: Env, flags: NamedTuple, num_episodes: int, current_board_feature_fncs: list, future_feature_fncs: list, device: torch.device) -> list:
    """Generate a list where each entry is a dictionary of features corresponding to a single transition

    Args:
        net (DRCNet): Trained DRC network used to generate transitions
        env (Env): Sokoban environment
        flags (NamedTuple): flag object
        num_episodes (int): number of episodes to run to generate the transitions
        current_board_feature_fncs (list): list of tuples of the form (feature_name, feature_fnc), where each feature_fnc extracts a discrete feature from the current state of the Sokoban board; this feature is then added to the episode entry (dictionary) with the key feature_name
        future_feature_fncs (list): list of functions where each function adds a feature to the current transition corresponding to the value taken by some other feature in a future transition
        prob_accept (float): probability that each transition entry is independently accepted into the dataset

    Returns:
        list: returns probing_data, a list of dictionaries where each dictionary contains features for a single transition generated by the DRC agent
    """

    rnn_state = net.initial_state(batch_size=1, device=device)
    state = env.reset() 
    state = {"real_states": torch.tensor(state).permute(2,0,1).unsqueeze(0)}
    env_out = util.init_env_out(state, flags, dim_actions=1, tuple_action=False)

    episode_length = 0
    board_num = 0
    probing_data = []
    episode_entry = []

    actor_out, rnn_state = net(env_out, rnn_state, greedy=True)
    trans_entry = {feature:fnc(state["real_states"][0]) for feature, fnc in current_board_feature_fncs}
    trans_entry["action"] = actor_out.action.item()
    trans_entry["value"] = round(actor_out.baseline.item(), 3) 
    trans_entry["board_state"] = state["real_states"][0].detach().cpu() 
    trans_entry["hidden_states"] = net.hidden_state[0].detach().cpu()
    trans_entry["board_num"] = board_num
    trans_entry["level"] = env.level
    episode_length += 1

    while(board_num < num_episodes):

        state, reward, done, info = env.step(actor_out.action)
        state = {"real_states": torch.tensor(state).permute(2,0,1).unsqueeze(0)}
        episode_entry.append(trans_entry)

        if episode_length > 1 and episode_entry[-1]["level"] != env.level:
            #print(env.level, episode_entry[-1]["level"], len(episode_entry))
            level_data = episode_entry
            for fnc in future_feature_fncs:
                episode_entry = fnc(level_data)
            probing_data += level_data
            episode_entry = []
    
        if done:
            #print("done", env.level, episode_entry[-1]["level"], len(episode_entry), episode_entry[-1].keys())
            for fnc in future_feature_fncs:
                episode_entry = fnc(episode_entry)
            for trans_idx, trans_entry in enumerate(episode_entry):
                trans_entry["steps_remaining"] = episode_length - trans_idx
                trans_entry["steps_taken"] = trans_idx+1
            
            probing_data += episode_entry

            episode_length = 0
            board_num += 1
            #print("Data collected from episode", board_num, "with episode length of", len(episode_entry))
            episode_entry = []
            rnn_state = net.initial_state(batch_size=1, device=device)
            state = env.reset()
            state = {"real_states": torch.tensor(state).permute(2,0,1).unsqueeze(0)}

        env_out = util.create_env_out(actor_out.action, state, torch.tensor([reward]), torch.tensor([done]), info, flags)
        actor_out, rnn_state = net(env_out, rnn_state, greedy=True)

        trans_entry = {feature:fnc(state["real_states"][0]) for feature, fnc in current_board_feature_fncs}
        trans_entry["action"] = actor_out.action.item()
        trans_entry["value"] = round(actor_out.baseline.item(), 3) 
        trans_entry["board_state"] = state["real_states"][0].detach().cpu() 
        trans_entry["hidden_states"] = net.hidden_state[0].detach().cpu() 
        trans_entry["board_num"] = board_num
        trans_entry["level"] = env.level
        episode_length += 1

    return probing_data


class ProbingDataset(Dataset):
    def __init__(self, data: list):
        self.data = data
    def __len__(self) -> int:
        return len(self.data)
    def __getitem__(self, index: int) -> dict:
        return self.data[index]
    def get_feature_range(self, feature: str) -> tuple[int, int]:
        assert feature in self.data[0].keys(), f"Please enter a feature in dataset: {self.data[0].keys()}"
        min_feature_value, max_feature_value = self.data[0][feature], self.data[0][feature]
        for entry in self.data:
            if entry[feature] > max_feature_value:
                max_feature_value = entry[feature]
            elif entry[feature] < min_feature_value:
                min_feature_value = entry[feature]
        return (min_feature_value, max_feature_value)


class ProbingDatasetCleaned(Dataset):
    def __init__(self, data: list):
        self.data = data
    def __len__(self) -> int:
        return len(self.data)
    def __getitem__(self, index: int) -> tuple:
        return self.data[index]
        

if __name__=="__main__":

    parser = argparse.ArgumentParser(description="create probing dataset")
    parser.add_argument("--num_episodes", type=int, default=1000, help="number of episodes to collect data from")
    parser.add_argument("--model_name", type=str, default="250m", help="name of agent checkpoint on which to run experiments")
    parser.add_argument("--seed", type=int, default=0, help="env seed")
    parser.add_argument("--name", type=str, default="train", help="name of dataset to create")
    parser.add_argument("--num_layers", type=int, default=3, help="number of convlstm layers the agent has")
    parser.add_argument("--num_ticks", type=int, default=3, help="number of internal ticks the agent performs")
    parser.add_argument('--resnet', action='store_true')
    parser.add_argument('--only_solved', action='store_true')
    args = parser.parse_args()

    device = torch.device("cpu") 

    env = thinker.make(
        f"gym_pilleater/PillEater-v0", 
        env_n=1, 
        gpu= False,
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
    if args.resnet:
        net = ResNet(
            obs_space=env.observation_space,
            action_space=env.action_space,
            flags=flags,
            record_state=True,
            num_layers=args.num_layers,
            input_dim=14
            )
    else:
        net = DRCNet(
            obs_space=env.observation_space,
            action_space=env.action_space,
            flags=flags,
            record_state=True,
            num_ticks=args.num_ticks,
            num_layers=args.num_layers,
            input_dim=14
        )
    ckp_path = "../../checkpoints/pilleater"
    ckp_path = os.path.join(util.full_path(ckp_path), f"ckp_actor_realstep{args.model_name}.tar")

    ckp = torch.load(ckp_path, env.device, weights_only=False)
    net.load_state_dict(ckp["actor_net_state_dict"], strict=False)
    net.to(env.device)
    net.eval()

    env = gym.make("gym_pilleater/PillEater-v0")
    env.seed(args.seed)

    agent_loc_detector = make_current_board_feature_detector(feature_idxs=[13], mode="loc")
    spooky_ghost_loc_detector = make_current_board_feature_detector(feature_idxs=[2,6,10], mode="loc")
    nonspooky_ghost_loc_detector = make_current_board_feature_detector(feature_idxs=[3,4,7,8,11,12], mode="loc")
    ghost_loc_detector = make_current_board_feature_detector(feature_idxs=[2,3,4,6,7,8,10,11,12], mode="loc")

    current_board_feature_fncs = [("agent_loc", agent_loc_detector),
                                  ("ghost_loc", ghost_loc_detector),
                                  ("spooky_ghost_loc", spooky_ghost_loc_detector),
                                  ("nonspooky_ghost_loc", nonspooky_ghost_loc_detector)]
    
    future_feature_fncs = [
                        make_agent_info_extractor(ahead=3),
                        make_agent_info_extractor(ahead=6),
                        make_agent_info_extractor(ahead=8),
                        make_agent_info_extractor(ahead=10),
                        make_agent_info_extractor(ahead=12),
                        make_agent_info_extractor(ahead=600),
                        make_agent_info_extractor(ahead=16),
                        make_agent_info_extractor(ahead=32),
                        make_agent_info_extractor(ahead=24),
                        ]
    
    future_feature_fncs += [make_trajectory_detector(feature_name="ghost_loc", steps_ahead=i) for i in [3,6,16]]
    future_feature_fncs += [make_trajectory_detector(feature_name="spooky_ghost_loc", steps_ahead=i) for i in [3,6,16]]
    future_feature_fncs += [make_trajectory_detector(feature_name="nonspooky_ghost_loc", steps_ahead=i) for i in [3,6,16]]

    probing_data = create_probing_data(
                                        net=net,
                                        env=env,
                                        flags=flags,
                                        num_episodes=args.num_episodes,
                                        current_board_feature_fncs=current_board_feature_fncs,
                                        future_feature_fncs=future_feature_fncs,
                                        device=device
                                       )

    print(f"Dataset {args.name}_data_full_{args.model_name} contains {len(probing_data)} transitions")
    
    if not os.path.exists("./data"):
        os.mkdir("./data")
    torch.save(ProbingDataset(probing_data), f"./data/{args.name}_data_full_{args.model_name}" + ("_resnet" if args.resnet else "") + ".pt")
 

