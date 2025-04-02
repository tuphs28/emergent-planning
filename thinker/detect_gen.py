from thinker.util import __project__
import os
import argparse
import yaml
import numpy as np
from collections import deque
import torch
from thinker.actor_net import ActorNet
from thinker.main import Env
import thinker.util as util
from thinker.util import init_env_out, create_env_out


class DetectBuffer:
    def __init__(self, outdir, t, rec_t, logger, delay_n=5):
        """
        Store training data grouped in planning stages and output
        whenever the target output is also readydd
            Args:
                N (int): number of planning stage per training output
                delay_n (int): number of planning stage delayed in the output y
                rec_t (int): number of step in a planning stage
                K (int): number of block to merge into
        """
        self.outdir = outdir
        self.t = t # number of time step per file
        self.rec_t = rec_t
        self.logger = logger        
        self.delay_n = delay_n        

        self.processed_n, self.xs, self.ys, self.done, self.step_status = 0, [], [], [], []
        self.file_idx = -1
    
    def insert(self, xs, ys, done, step_status):
        """
        Args:
            xs (dict): dict of training input, with each elem having the
                shape of (B, *)            
            ys (dict): dict of target output, each with shape (B, N) or (B,); 
                item starts with "cost" will have or done across all deplay_n steps and is assumed to be bool
            done (tensor): bool tensor of shape (B), being the indicator of episode end
            step_status (int): int indicating current step status
        Output:
            save train_xs in shape (N, rec_t, B, *) and train_y in shape (N, B)
        """
        #print("data received! ", y.shape, id, cur_t)
        last_step_real = (step_status == 0) | (step_status == 3)
        if len(self.step_status) == 0 and not last_step_real: return self.file_idx  # skip until real step
                
        self.xs.append(util.dict_map(xs, lambda x:x.cpu()))
        if ys is not None:
            self.ys.append(util.dict_map(ys, lambda y:y.cpu()))
        else:
            ys = dict({k:torch.zeros_like(v) for k, v in self.ys[0].items()})
            self.ys.append(ys)
        self.done.append(done.cpu())
        self.step_status.append(step_status)
        self.processed_n += int(last_step_real)

        if (self.processed_n >= self.t + self.delay_n + 1):               
            self.file_idx += 1                     
            out = self._extract_data(self.t)
            self.processed_n = sum([int(i == 0) + int(i == 3) for i in self.step_status])
            assert self.processed_n == self.delay_n+1, f"should only have {self.delay_n + 1} data left instead of {self.processed_n}"
            path = f'{self.outdir}/data_{self.file_idx}.pt'
            torch.save(out, path)
            out_shape = out[0]['env_state'].shape
            n = self.file_idx * out_shape[0] * out_shape[2]
            self.logger.info(f"{n}: File saved to {path}; env_state shape {out_shape}")

        return self.file_idx   

    def _extract_data(self, t):
        # obtain the first N planning stage and the corresponding target_y in data
        xs, ys, done, step_status = self._collect_data(t)
        future_ys, future_done = self._collect_data(self.delay_n, y_done_only=True)
        for k in self.ys[0].keys():
            ys[k] = torch.concat([ys[k], future_ys[k]], dim=0)        
        done = torch.concat([done, future_done], dim=0)                
        
        last_step_real = (step_status == 0) | (step_status == 3)
        assert last_step_real[0], "cur_t should start with 0"
        assert last_step_real.shape[0] == t*self.rec_t, \
            f" last_step_real.shape is {last_step_real.shape}, expected {t*self.rec_t} for the first dimension."        
        for k, v in ys.items():
            assert v.shape[0] == (t + self.delay_n)*self.rec_t, \
                f" ys[{k}].shape is {v.shape}, expected {(t + self.delay_n)*self.rec_t} for the first dimension."        
            B = v.shape[1]
            ys[k] = ys[k].view((t + self.delay_n, self.rec_t, B)+ys[k].shape[2:])[:, 0]

        done = done.view(t + self.delay_n, self.rec_t, B)[:, 0]
        step_status = step_status.view(t, self.rec_t)
        # compute target_y
        target_ys = self._compute_target_y(ys, done, self.delay_n)

        for k in xs.keys():
            xs[k] = xs[k].view((t, self.rec_t) + xs[k].shape[1:])
        
        xs["done"] = done[:t]
        xs["step_status"] = step_status
                
        return xs, target_ys

    def _collect_data(self, t, y_done_only=False):
        # collect the first t stage from data
        step_status = torch.tensor(self.step_status, dtype=torch.long)
        next_step_real = (step_status == 2) | (step_status == 3)        
        idx = torch.nonzero(next_step_real, as_tuple=False).squeeze()    
        last_idx = idx[t-1] + 1
        ys = {}
        for k in self.ys[0].keys():
            ys[k] = torch.stack([v[k] for v in self.ys[:last_idx]], dim=0)                
        done = torch.stack(self.done[:last_idx], dim=0)
        if not y_done_only:
            xs = {}
            for k in self.xs[0].keys():
                xs[k] = torch.stack([v[k] for v in self.xs[:last_idx]], dim=0)                
            step_status = step_status[:last_idx]
            self.xs = self.xs[last_idx:]
            self.ys = self.ys[last_idx:]
            self.done = self.done[last_idx:]
            self.step_status = self.step_status[last_idx:]
            return xs, ys, done, step_status
        else:
            return ys, done
        
    def _compute_target_y(self, ys, done, delay_n):        
        # target_y[i] = (y[i] | (~done[i+1] & y[i+1]) | (~done[i+1] & ~done[i+2] & y[i+2]) | ... | (~done[i+1] & ~done[i+2] & ... & ~done[i+M] & y[i+M]))
        target_ys = {}
        for k, y in ys.items():
            t, b = y.shape[:2]
            t = t - delay_n
            not_done_cum = torch.ones(delay_n, t, b, dtype=bool)
            target_y = y.clone()[:-delay_n]
            not_done_cum[0] = ~done[1:1+t]            
            for m in range(1, delay_n):
                not_done_cum[m] = not_done_cum[m-1] & ~done[m+1:m+1+t]
            if k.startswith("cost"):
                target_y = target_y | (not_done_cum[0] & y[1:1+t])
                assert y.dtype == torch.bool
                for m in range(1, delay_n):                
                    target_y = target_y | (not_done_cum[m] & y[m+1:m+1+t])
            else:
                target_y = y[delay_n:delay_n+t]
                if k.startswith("last_real_actions"):
                    for m in range(0, delay_n):                
                        target_y[..., m][~not_done_cum[m]] = 0 
                else:                    
                    target_y[~not_done_cum[delay_n-1]] = 0
            target_ys[k] = target_y
        return target_ys


def detect_gen(total_n, env_n, delay_n, greedy, confuse, no_reset, skip_im, savedir, outdir, xpid, suffix="", n=0):

    _logger = util.logger()
    _logger.info(f"Initializing {xpid} from {savedir}")
    device = torch.device("cuda")

    ckpdir = os.path.join(savedir, xpid)     
    if os.path.islink(ckpdir): ckpdir = os.readlink(ckpdir)  
    ckpdir =  os.path.abspath(os.path.expanduser(ckpdir))
    outdir = os.path.abspath(os.path.expanduser(outdir))

    config_path = os.path.join(ckpdir, 'config_c.yaml')
    flags = util.create_flags(config_path, save_flags=False)
    flags.shallow_enc = False

    env = Env(
            name=flags.name,
            env_n=env_n,
            gpu=True,
            train_model=False,
            parallel=False,
            savedir=savedir,        
            xpid=xpid,
            ckp=True,
            return_x=True,
            return_h=True,
            rand_action_eps=0,
        )

    disable_thinker = flags.wrapper_type == 1   
    im_rollout = disable_thinker and env.has_model and not skip_im
    mcts = getattr(flags, "mcts", False)

    obs_space = env.observation_space
    action_space = env.action_space 

    actor_param = {
        "obs_space": obs_space,
        "action_space": action_space,
        "flags": flags,
        "tree_rep_meaning": env.get_tree_rep_meaning() if not disable_thinker else None,
        "record_state": True,
    }
    actor_net = ActorNet(**actor_param)

    if not mcts:
        path = os.path.join(ckpdir, "ckp_actor.tar")
        checkpoint = torch.load(path, map_location=torch.device("cpu"))
        actor_net.set_weights(checkpoint["actor_net_state_dict"], strict=False)
        actor_net.to(device)
        actor_net.train(False)

    state = env.reset()
    env_out = init_env_out(state, flags=flags, dim_actions=actor_net.dim_actions, tuple_action=actor_net.tuple_action)  
    last_real_state = env_out.real_states[-1]
    actor_state = actor_net.initial_state(batch_size=env_n, device=device)

    file_idx = 0

    # create dir
    if suffix: suffix = "-" + suffix
    while True:
        name = "%s%s-%d" % (xpid, suffix, n)
        outdir_ = os.path.join(outdir, name)
        if not os.path.exists(outdir_):
            os.makedirs(outdir_)
            print(f"Outputting to {outdir_}")
            break
        n += 1
    outdir = outdir_

    rec_t=flags.rec_t if not im_rollout and not mcts else delay_n + 1
    if skip_im: rec_t = 1
    detect_buffer = DetectBuffer(outdir=outdir, t=12800//env_n, rec_t=rec_t, logger=_logger, delay_n=delay_n)
    file_n = total_n // (env_n * detect_buffer.t) + 3
    _logger.info(f"Data output directory: {outdir}")
    _logger.info(f"Number of file to be generated: {file_n}")

    with torch.set_grad_enabled(False):
        
        actor_out, actor_state = actor_net(env_out=env_out, core_state=actor_state, greedy=greedy)            
        if not disable_thinker:
            primary_action, reset_action = actor_out.action
        else:
            primary_action, reset_action = actor_out.action, None

        # save setting
        env_state_shape = env.observation_space["real_states"].shape[1:]
        #if rescale: env_state_shape = (3, 40, 40)
        tree_rep_shape = env.observation_space["tree_reps"].shape[1:] if not disable_thinker else None
        hidden_state_shape = actor_net.hidden_state.shape[1:] if disable_thinker else None

        flags_detect = {
            "dim_actions": actor_net.dim_actions,
            "num_actions": actor_net.num_actions,
            "tuple_actions": actor_net.tuple_action,
            "name": flags.name,
            "env_state_shape": list(env_state_shape),
            "tree_rep_shape": list(tree_rep_shape) if not disable_thinker else None,
            "hidden_state_shape": list(hidden_state_shape) if disable_thinker else None,
            "rec_t": flags.rec_t,
            "model_decoder_depth": getattr(flags, "model_decoder_depth", 0),
            "delay_n": delay_n,
            "ckpdir": ckpdir,
            "xpid": xpid,        
            "dxpid": name,
            "disable_thinker": disable_thinker,
            "im_rollout": im_rollout,      
            "mcts": mcts,
        }

        yaml_file_path = os.path.join(outdir, 'config_detect.yaml')
        with open(yaml_file_path, 'w') as file:
            yaml.dump(flags_detect, file)

        rets = []
        last_file_idx = None
        last_real_actions = torch.zeros(primary_action.shape+(delay_n,), 
                                        dtype=primary_action.dtype, 
                                        device=device
                                        )
        last_im_actions_buffer = deque([], maxlen=delay_n)
        accs = [[] for _ in range(delay_n)]

        last_dones = torch.zeros(env_n, delay_n, dtype=torch.bool, device=device)
        done = torch.zeros(env_n, dtype=torch.bool, device=device)
        print_n = 0        

        if confuse: confuse_add = util.ConfuseAdd(device)
                
        while(True):
            state, reward, done, info = env.step(
                primary_action=primary_action, 
                reset_action=reset_action if not no_reset else torch.zeros_like(reset_action), 
                action_prob=actor_out.action_prob[-1])    
            
            env_out = create_env_out(actor_out.action, state, reward, done, info, flags=flags)
            if torch.any(done):
                rets.extend(info["episode_return"][done].cpu().tolist())

            done = done 
            step_status = info['step_status'][0].item() if not im_rollout else 0       
            last_step_real =  step_status in [0, 3]
            next_step_real =  step_status in [2, 3]            
            
            if last_step_real or im_rollout:
                last_real_actions = torch.cat([last_real_actions[..., 1:], primary_action.unsqueeze(-1)], dim=-1)
                last_dones = torch.cat([last_dones[..., 1:], done.unsqueeze(-1)], dim=-1)
                last_real_state = env_out.real_states[-1]
            
            last_actor_state = actor_state
            actor_out, actor_state = actor_net(env_out=env_out, core_state=actor_state, greedy=greedy)            
            if not disable_thinker:
                primary_action, reset_action = actor_out.action
            else:
                primary_action, reset_action = actor_out.action, None
            
            ys = {"cost": info['cost'],
                  "last_real_actions": last_real_actions,
                  "last_dones": last_dones,
                  }
            if im_rollout and len(last_im_actions_buffer) == delay_n:
                for t in range(delay_n):
                    acc = torch.all(last_real_actions[:, :t+1] == last_im_actions_buffer[0][:, :t+1], dim=-1)
                    acc = torch.mean(acc.float()).item()
                    accs[t].append(acc)
            
            # write to detect buffer
            if not disable_thinker:
                env_state = env_out.xs[0] 
            else:
                env_state = env_out.real_states[0]
                env_state = env.normalize(env_state)
            
            env_state = env_state.half()
            xs = {
                "env_state": env_state,
                "pri_action": primary_action,            
                "cost": info["cost"],
            }
            if not disable_thinker and getattr(flags, "model_decoder_depth", 0) > 0:
                xs["real_state"] = last_real_state # abstract model; need to separate store the first real state

            if not disable_thinker:
                if not mcts: xs.update({"tree_rep": state["tree_reps"]})
                xs.update({"reset_action": actor_out.action[1]})
            else:
                if disable_thinker:
                    xs.update({
                        "hidden_state": actor_net.hidden_state
                    })                          

            if not (mcts and step_status != 0) and not (skip_im and not last_step_real): # no recording for non-real mcts
                if skip_im: step_status = 3
                file_idx = detect_buffer.insert(xs, ys, done, step_status)

            if im_rollout or (mcts and next_step_real):
                # generate imaginary rollout or most visited rollout (mcts)
                action = env_out.last_pri
                model_net_out = env.model_net(
                    env_state=env_out.real_states[0],
                    done=env_out.done[0],
                    actions=action,
                    state=None,
                )
                new_env_out = env_out
                im_actor_state = last_actor_state
                if mcts:
                    most_visited_actions = torch.tensor(env.most_visited_path(delay_n), dtype=torch.long, device=device)
                last_im_actions = []
                for m in range(delay_n):
                    if not mcts:
                        actor_out, im_actor_state = actor_net(env_out=new_env_out, core_state=im_actor_state, greedy=greedy)   
                        action = actor_out.action
                    else:
                        action = most_visited_actions[m]
                    last_im_actions.append(action)
                    model_net_out = env.model_net.forward_single(
                        state=model_net_out.state,
                        action=action,
                    )
                    xs = model_net_out.xs                    
                    if confuse: xs = confuse_add.add(xs)
                    new_state = {"real_states": (torch.clamp(xs,0,1)*255).to(torch.uint8)[0]}
                    new_env_out = create_env_out(action, new_state, reward, done, info, flags=flags)
                    xs = {
                        "env_state": xs[0].half(),
                        "pri_action": action,            
                        "cost": torch.zeros_like(info["cost"]),
                    }
                    if im_rollout: xs["hidden_state"] = actor_net.hidden_state
                    if mcts: xs["reset_action"] = torch.zeros_like(actor_out.action[1])
                    file_idx = detect_buffer.insert(
                        xs, 
                        None, 
                        torch.zeros_like(done), 
                        1 if m < delay_n - 1 else 2
                    )
                last_im_actions = torch.stack(last_im_actions, dim=-1)
                last_im_actions_buffer.append(last_im_actions)
            
            if file_idx >= file_n: 
                # last file is for validation
                os.rename(f'{outdir}/data_{file_idx}.pt', f'{outdir}/val.pt')
                os.rename(f'{outdir}/data_{file_idx-1}.pt', f'{outdir}/test.pt')
                break   
            if print_n % 100 == 0 and len(rets) > 0:
                print_str = f"{print_n} - Episode {len(rets)}; Return {np.mean(np.array(rets)):.2f}"
                if len(accs) > 0: 
                    print_str += f" im-rollout acc "
                    for t in range(delay_n):
                        print_str += f"{np.mean(np.array(accs[t])):.2f} "
                print(print_str)
            
            print_n += 1
            last_file_idx = file_idx
        
        env.close()
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=f"Thinker data generalization")
    parser.add_argument("--outdir", default="../data/transition", help="Output directory.")
    parser.add_argument("--savedir", default="../logs/__project__", help="Checkpoint directory.")
    parser.add_argument("--xpid", default="latest", help="id of the run.")    
    parser.add_argument("--project", default="", help="project of the run.")  
    parser.add_argument("--total_n", default=100000, type=int, help="Number of real steps.")
    parser.add_argument("--env_n", default=128, type=int, help="Batch size in generation.")
    parser.add_argument("--delay_n", default=5, type=int, help="Delay step in predicting danger.")
    parser.add_argument("--greedy", action="store_true", help="Use greedy policy.")
    parser.add_argument("--no_reset", action="store_true", help="Force no resetting.")
    parser.add_argument("--confuse", action="store_true", help="Add confuse image.")
    parser.add_argument("--skip_im",  action="store_true", help="Skip im state.")
    parser.add_argument("--suffix", default="", help="Suffix of the data id.")
    parser.add_argument("--n", default=0, type=int, help="Numer of the data id.")

    flags = parser.parse_args()    
    project = flags.project if flags.project else __project__
    
    flags.outdir=flags.outdir.replace("__project__", project)
    flags.savedir=flags.savedir.replace("__project__", project)

    detect_gen(
        total_n=flags.total_n,
        env_n=flags.env_n,
        delay_n=flags.delay_n,
        greedy=flags.greedy,        
        confuse=flags.confuse,
        no_reset=flags.no_reset,
        skip_im=flags.skip_im,
        savedir=flags.savedir,
        outdir=flags.outdir,
        xpid=flags.xpid,
        suffix=flags.suffix, 
        n=flags.n,
    )