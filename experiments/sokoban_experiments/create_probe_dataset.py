import sys
sys.path.append('/home/tom/mlmi/project/testrepo/sokoban')
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

def make_current_board_feature_detector(feature_idxs: list, mode: str) -> Callable:
    """Create feature detector functions to extract discrete features from mini-sokoban boards. Boards must be (7,8,8) arrays

    Args:
        feature_idxs (list): list index of feature of interest (see sokoban.cpp);
        mode (str): type of feature detector to construct: "adj" (to count number of adjacent features), "num" (to count total number of features on board) or "loc" (extract location of features)

    Returns:
        Callable: feature detector function, takes in a board state and returns the desired feature
    """
    if mode == "adj":
        def feature_detector(board: torch.tensor) -> int:
            h, w = board.shape[1:]
            x, y = ((board[4,:,:]==1) + (board[5,:,:]==1)).nonzero()[0,:]
            adj_coords = [(xp, yp) for xp, yp in [(x+1,y), (x-1,y), (x,y+1), (x,y-1)] if xp>-1 and xp<h and yp>-1 and yp<w]
            n_hits = 0
            for (xp,yp) in adj_coords:
                for feature_idx in feature_idxs:
                    if board[feature_idx, xp, yp] == 1:
                        n_hits += 1
            return n_hits
    elif mode == "num":
        def feature_detector(board: torch.tensor) -> int:
            return sum([torch.sum((board[feature_idx,:,:]==1).int()) for feature_idx in feature_idxs]).item()
    elif mode == "loc":
        def feature_detector(board):
            locs_xy = sum([(board[feature_idx,:,:]==1) for feature_idx in feature_idxs]).nonzero()
            locs = tuple([(8*x+y).item() for (x,y) in locs_xy]) # each location is an int in range [0,63]
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


def generate_aug_trans(episode_entry):
    trans = episode_entry[-1]
    agent_loc = trans["agent_loc"][0]
    agent_loc = ((agent_loc -(agent_loc % 8))//8, agent_loc % 8,)
    agent_y, agent_x = agent_loc
    box_locs = [((box_loc -(box_loc % 8))//8, box_loc % 8) for box_loc in trans["tracked_box_loc"]]
    wall_locs = [((wall_loc-(wall_loc % 8))//8, wall_loc % 8) for wall_loc in trans["board_state"][0].view(-1).topk(k=(trans["board_state"][0]==1).to(int).sum()).indices]
    if ((agent_y-1,agent_x) in box_locs) and trans["action"] == 1:
        if agent_y > 1 and (agent_y-2,agent_x) not in wall_locs and (agent_y-2,agent_x) not in box_locs:
            new_box_locs = [box_loc if box_loc!=(agent_y-1,agent_x) else (agent_y-2,agent_x) for box_loc in box_locs]
            new_agent_loc = (agent_y-1,agent_x)
        else:
            new_agent_loc = agent_loc
            new_box_locs = box_locs
    elif ((agent_y+1,agent_x) in box_locs) and trans["action"] == 2:
        if agent_y < 6 and (agent_y+2,agent_x) not in wall_locs and (agent_y+2,agent_x) not in box_locs:
            new_box_locs = [box_loc if box_loc!=(agent_y+1,agent_x) else (agent_y+2,agent_x) for box_loc in box_locs]
            new_agent_loc = (agent_y+1,agent_x)
        else:
            new_agent_loc = agent_loc
            new_box_locs = box_locs
    elif ((agent_y,agent_x-1) in box_locs) and trans["action"] == 3:
        if agent_x > 1  and (agent_y,agent_x-2) not in wall_locs and (agent_y,agent_x-2) not in box_locs:
            new_box_locs = [box_loc if box_loc!=(agent_y,agent_x-1) else (agent_y,agent_x-2) for box_loc in box_locs]
            new_agent_loc = (agent_y,agent_x-1)
        else:
            new_box_locs = box_locs
            new_agent_loc = agent_loc
    elif ((agent_y,agent_x+1) in box_locs) and trans["action"] == 4:
        if agent_x < 6 and (agent_y,agent_x+2) not in wall_locs and (agent_y,agent_x+2) not in box_locs:
            new_box_locs = [box_loc if box_loc!=(agent_y,agent_x+1) else ((agent_y,agent_x+2) if agent_x<6 else (agent_y,agent_x+1)) for box_loc in box_locs]
            new_agent_loc = (agent_y,agent_x+1)
        else:
            new_box_locs = box_locs
            new_agent_loc = agent_loc
    else:
        new_box_locs = box_locs
        if trans["action"] == 1 and agent_y > 0:
            new_agent_loc = (agent_y-1,agent_x)
        elif trans["action"] == 2 and agent_y < 7:
            new_agent_loc = (agent_y+1,agent_x)
        elif trans["action"] == 3 and agent_x > 0:
            new_agent_loc = (agent_y,agent_x-1)
        elif trans["action"] == 4 and agent_y < 7:
            new_agent_loc = (agent_y, agent_x+1)
        else:
            new_agent_loc = agent_loc

    new_box_locs = tuple([(8*y+x) for y,x in new_box_locs])
    new_agent_loc = tuple([8*new_agent_loc[0] + new_agent_loc[1]])
    trans = {"tracked_box_loc": new_box_locs, "agent_loc": new_agent_loc, "action": 0}
    return trans

def make_agent_info_extractor() -> Callable:
    def agent_info_extractor(episode_entry: list) -> list:
        # track squares from which agent performs actions to leave
        aug_episode_entry = episode_entry + [generate_aug_trans(episode_entry)]
        for trans_idx, trans in enumerate(aug_episode_entry):
            board_locs = torch.zeros((8,8), dtype=int)
            for loc_idx in range(64):
                for future_trans_idx, future_trans in enumerate(aug_episode_entry[trans_idx:-1]):
                    if loc_idx in future_trans["agent_loc"] and (future_trans["agent_loc"] != aug_episode_entry[trans_idx+future_trans_idx+1]["agent_loc"]): #NB: ignore no-ops and effective no-ops since want action we leave square with 
                        board_locs[(loc_idx-loc_idx%8)//8, loc_idx%8] = future_trans["action"]
                        break
            trans["agent_onto_with"] = board_locs
            new_board_locs = torch.zeros((8,8), dtype=int)
            new_board_locs[board_locs != 0 ] = 1
            trans["agent_from"] = new_board_locs
        episode_entry = aug_episode_entry[:-1]
        # track squares from which agent performs action to enter
        aug_episode_entry = episode_entry + [generate_aug_trans(episode_entry)]
        for trans_idx, trans in enumerate(aug_episode_entry):
            board_locs = torch.zeros((8,8), dtype=int)
            for loc_idx in range(64):
                for future_trans_idx, future_trans in enumerate(aug_episode_entry[trans_idx+1:]):
                    if loc_idx in future_trans["agent_loc"] and aug_episode_entry[trans_idx+future_trans_idx]["agent_loc"] != future_trans["agent_loc"]:
                        board_locs[(loc_idx-loc_idx%8)//8, loc_idx%8] = aug_episode_entry[trans_idx+future_trans_idx]["action"]
                        break
            trans["agent_onto_after"] = board_locs
            new_board_locs = torch.zeros((8,8), dtype=int)
            new_board_locs[board_locs != 0 ] = 1
            trans["agent_onto"] = new_board_locs
        episode_entry = aug_episode_entry[:-1]
        for trans_idx, trans in enumerate(episode_entry):
            board_locs = torch.zeros((8,8), dtype=int)
            for loc_idx in range(64):
                for future_trans in episode_entry[trans_idx+1:]:
                    if loc_idx in future_trans["agent_loc"]:
                        board_locs[(loc_idx-loc_idx%8)//8, loc_idx%8] += (1 if board_locs[(loc_idx-loc_idx%8)//8, loc_idx%8] <= 2 else 0)
            trans["agent_loc_count"] = board_locs
        return episode_entry
    return agent_info_extractor

def make_box_info_extractor() -> Callable:
    def box_info_extractor(episode_entry: list) -> list:
        # track box_locs
        tracked_box_locs = [episode_entry[0]["box_loc"][i] for i in range(4)]
        episode_entry[0]["tracked_box_loc"] = tuple([tracked_box_locs[i] for i in range(4)])
        for trans in episode_entry[1:]:
            trans_box_locs = trans["box_loc"]
            for i in range(4):
                if tracked_box_locs[i] not in trans_box_locs:
                    for j in range(4):
                        if trans_box_locs[j] not in tracked_box_locs:
                            tracked_box_locs[i] = trans_box_locs[j]
            trans["tracked_box_loc"] = tuple([tracked_box_locs[i] for i in range(4)])
              
        # track locations where box is pushed *after* action X 
        episode_entry[0]["tracked_box_loc_change_after_action_1"] = tuple()
        episode_entry[0]["tracked_box_loc_change_after_action_2"] = tuple()
        episode_entry[0]["tracked_box_loc_change_after_action_3"] = tuple()
        episode_entry[0]["tracked_box_loc_change_after_action_4"] = tuple()
        aug_episode_entry = episode_entry + [generate_aug_trans(episode_entry)]
        for trans_idx, trans in enumerate(aug_episode_entry[1:]):
            trans["tracked_box_loc_change_after_action_1"] = tuple()
            trans["tracked_box_loc_change_after_action_2"] = tuple()
            trans["tracked_box_loc_change_after_action_3"] = tuple()
            trans["tracked_box_loc_change_after_action_4"] = tuple()
            if trans["tracked_box_loc"] != aug_episode_entry[trans_idx]["tracked_box_loc"]:
                for i in range(4):
                    if trans["tracked_box_loc"][i] != aug_episode_entry[trans_idx]["tracked_box_loc"][i]:
                        trans[f"tracked_box_loc_change_after_action_{aug_episode_entry[trans_idx]['action']}"] = tuple([trans["tracked_box_loc"][i]])

        # track locations where box is pushed *with* action X 
        aug_episode_entry[0]["tracked_box_loc_change_with_action_1"] = tuple()
        aug_episode_entry[0]["tracked_box_loc_change_with_action_2"] = tuple()
        aug_episode_entry[0]["tracked_box_loc_change_with_action_3"] = tuple()
        aug_episode_entry[0]["tracked_box_loc_change_with_action_4"] = tuple()
        for trans_idx, trans in enumerate(aug_episode_entry[1:]):
            trans["tracked_box_loc_change_with_action_1"] = tuple()
            trans["tracked_box_loc_change_with_action_2"] = tuple()
            trans["tracked_box_loc_change_with_action_3"] = tuple()
            trans["tracked_box_loc_change_with_action_4"] = tuple()
            if trans["tracked_box_loc"] != aug_episode_entry[trans_idx]["tracked_box_loc"]:
                for i in range(4):
                    if trans["tracked_box_loc"][i] != aug_episode_entry[trans_idx]["tracked_box_loc"][i]:
                        aug_episode_entry[trans_idx][f"tracked_box_loc_change_with_action_{aug_episode_entry[trans_idx]['action']}"] = tuple([aug_episode_entry[trans_idx]["tracked_box_loc"][i]])
                 
        # track the direction from which a box is next pushed onto this square (0 if no box is pushed onto this square for the rest of the episode)
        for trans_idx, trans in enumerate(aug_episode_entry):
            board_locs = torch.zeros((8,8), dtype=int)
            for loc_idx in range(64):
                for future_trans in aug_episode_entry[trans_idx+1:]:
                    if loc_idx in future_trans["tracked_box_loc_change_after_action_1"]:
                        board_locs[(loc_idx-loc_idx%8)//8, loc_idx%8] = 1
                        break
                    elif loc_idx in future_trans["tracked_box_loc_change_after_action_2"]:
                        board_locs[(loc_idx-loc_idx%8)//8, loc_idx%8] = 2
                        break
                    elif loc_idx in future_trans["tracked_box_loc_change_after_action_3"]:
                        board_locs[(loc_idx-loc_idx%8)//8, loc_idx%8] = 3
                        break
                    elif loc_idx in future_trans["tracked_box_loc_change_after_action_4"]:
                        board_locs[(loc_idx-loc_idx%8)//8, loc_idx%8] = 4
                        break
            trans["tracked_box_next_push_onto_after"] = board_locs
            new_board_locs = torch.zeros((8,8), dtype=int)
            new_board_locs[board_locs != 0 ] = 1
            trans["tracked_box_next_push_onto"] = new_board_locs

        # track the direction in which a box is next pushed from this square (0 if no box is pushed onto this square for the rest of the episode)
        for trans_idx, trans in enumerate(aug_episode_entry):
            board_locs = torch.zeros((8,8), dtype=int)
            for loc_idx in range(64):
                for future_trans in aug_episode_entry[trans_idx:]:
                    if loc_idx in future_trans["tracked_box_loc_change_with_action_1"]:
                        board_locs[(loc_idx-loc_idx%8)//8, loc_idx%8] = 1
                        break
                    elif loc_idx in future_trans["tracked_box_loc_change_with_action_2"]:
                        board_locs[(loc_idx-loc_idx%8)//8, loc_idx%8] = 2
                        break
                    elif loc_idx in future_trans["tracked_box_loc_change_with_action_3"]:
                        board_locs[(loc_idx-loc_idx%8)//8, loc_idx%8] = 3
                        break
                    elif loc_idx in future_trans["tracked_box_loc_change_with_action_4"]:
                        board_locs[(loc_idx-loc_idx%8)//8, loc_idx%8] = 4
                        break
            trans["tracked_box_next_push_onto_with"] = board_locs
            new_board_locs = torch.zeros((8,8), dtype=int)
            new_board_locs[board_locs != 0 ] = 1
            trans["tracked_box_next_push_from"] = new_board_locs
        episode_entry = aug_episode_entry[:-1]
        return episode_entry
    return box_info_extractor



@torch.no_grad()
def create_probing_data(net: DRCNet, env: Env, flags: NamedTuple, num_episodes: int, current_board_feature_fncs: list, future_feature_fncs: list) -> list:
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

    rnn_state = net.initial_state(batch_size=1, device=env.device)
    state = env.reset() 
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
    print(trans_entry["hidden_states"].shape)
    trans_entry["board_num"] = board_num
    episode_length += 1

    while(board_num < num_episodes):

        state, reward, done, info = env.step(actor_out.action)
        trans_entry["reward"] = round(reward.item(), 3) # round rewards to 3 d.p.
        episode_entry.append(trans_entry)

        if done:
            for fnc in future_feature_fncs:
                episode_entry = fnc(episode_entry)
            for trans_idx, trans_entry in enumerate(episode_entry):
                trans_entry["steps_remaining"] = episode_length - trans_idx
                trans_entry["steps_taken"] = trans_idx+1
                trans_entry["return"] = sum([(0.97**t)*future_trans["reward"] for t, future_trans in enumerate(episode_entry[trans_idx:])])
 
            probing_data += episode_entry
            
            episode_length = 0
            board_num += 1
            print("Data collected from episode", board_num, "with episode length of", len(episode_entry))
            episode_entry = []
            rnn_state = net.initial_state(batch_size=1, device=env.device)

        env_out = util.create_env_out(actor_out.action, state, reward, done, info, flags)
        actor_out, rnn_state = net(env_out, rnn_state, greedy=True)

        trans_entry = {feature:fnc(state["real_states"][0]) for feature, fnc in current_board_feature_fncs}
        trans_entry["action"] = actor_out.action.item()
        trans_entry["value"] = round(actor_out.baseline.item(), 3) 
        trans_entry["board_state"] = state["real_states"][0].detach().cpu() 
        trans_entry["hidden_states"] = net.hidden_state[0].detach().cpu() 
        trans_entry["board_num"] = board_num
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
    parser.add_argument("--env_name", type=str, default="", help="level dataset to collect transitions from")
    parser.add_argument("--name", type=str, default="train", help="name of dataset to create")
    parser.add_argument("--num_layers", type=int, default=3, help="number of convlstm layers the agent has")
    parser.add_argument("--num_ticks", type=int, default=3, help="number of internal ticks the agent performs")
    parser.add_argument('--resnet', action='store_true')
    parser.add_argument('--only_solved', action='store_true')
    args = parser.parse_args()

    if torch.cuda.is_available(): 
        device = torch.device("cuda")
    else: 
        device = torch.device("cpu") 

    env = thinker.make(
        f"Sokoban-{args.env_name}v0", 
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
            num_layers=args.num_layers
            )
    else:
        net = DRCNet(
            obs_space=env.observation_space,
            action_space=env.action_space,
            flags=flags,
            record_state=True,
            num_ticks=args.num_ticks,
            num_layers=args.num_layers
        )
    ckp_path = "../../checkpoints/sokoban"
    ckp_path = os.path.join(util.full_path(ckp_path), f"ckp_actor_realstep{args.model_name}.tar")

    agent_loc_detector = make_current_board_feature_detector(feature_idxs=[4,5], mode="loc")
    box_loc_detector = make_current_board_feature_detector(feature_idxs=[2,3], mode="loc")
    tar_loc_detector = make_current_board_feature_detector(feature_idxs=[3,5,6], mode="loc")
    justtar_loc_detector = make_current_board_feature_detector(feature_idxs=[5,6], mode="loc")

    ckp = torch.load(ckp_path, map_location=env.device, weights_only=False)
    net.load_state_dict(ckp["actor_net_state_dict"], strict=False)
    net.to(env.device)
    net.eval()

    current_board_feature_fncs = [
        ("agent_loc", agent_loc_detector),
        ("box_loc", box_loc_detector),
        ("tar_loc", tar_loc_detector),
        ("justtar_loc", justtar_loc_detector),
    ]
    future_feature_fncs = [make_future_feature_detector(feature_name="action",steps_ahead=t, mode="ahead") for t in range(1,11)]
    future_feature_fncs += [make_box_info_extractor(), make_agent_info_extractor()]

    probing_data = create_probing_data(
                                        net=net,
                                        env=env,
                                        flags=flags,
                                        num_episodes=args.num_episodes,
                                        current_board_feature_fncs=current_board_feature_fncs,
                                        future_feature_fncs=future_feature_fncs,
                                       )

    print(f"Dataset {args.name}_data_full_{args.model_name} contains {len(probing_data)} transitions")
    
    if not os.path.exists("./data"):
        os.mkdir("./data")
    torch.save(ProbingDataset(probing_data), f"./data/{args.name}_data_full_{args.model_name}" + ("_resnet" if args.resnet else "") + ".pt")
 

