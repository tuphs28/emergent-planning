import torch
import torch.nn as nn
import numpy as np
import thinker
import thinker.util as util
import gym
import gym_sokoban
import pandas as pd
import numpy as np
from thinker.actor_net import DRCNet, ResNet
from train_conv_probe import ConvProbe
import os
from thinker.actor_net import sample
from thinker.util import EnvOut
from typing import Optional
import argparse

class ProbeIntervDRCNet:
    """
    Wrapper around DRCNet for probe interventions.
    """

    def __init__(self, drc_net, debug=False):
        self.drc_net = drc_net
        self.debug= debug

    def forward_normal(self, env_out, rnn_state):
        return self.drc_net(env_out, rnn_state)
    
    def forward_patch(self, env_out: EnvOut, rnn_state: tuple, greedy: bool = True,
                      activ_ticks: Optional[list] = None, 
                      patch_info: Optional[dict] = None):
    
        done = env_out.done
        T, B = done.shape
        x = self.drc_net.normalize(env_out.real_states.float())
        x = torch.flatten(x, 0, 1)
        x_enc = self.drc_net.encoder(x)
        core_input = x_enc.view(*((T, B) + x_enc.shape[1:]))

        assert len(core_input.shape) == 5
        core_output_list = []
        reset = done.float()
        if self.drc_net.record_state: 
            self.drc_net.core.hidden_state = []
            self.drc_net.core.hidden_state.append(torch.concat(rnn_state, dim=1)) 
        for n, (x_single, reset_single) in enumerate(
            zip(core_input.unbind(), reset.unbind())
        ):
            for t in range(self.drc_net.core.tran_t):

                if t > 0:
                    reset_single = torch.zeros_like(reset_single)
                reset_single = reset_single.view(-1)
                if len(patch_info.keys()) > 0:
                    if t in activ_ticks:
                        if self.debug:
                            print(f"----- patching activations for tick {t} ---- ")
                        output, rnn_state = self.forward_single_patch(
                            x=x_single,
                            core_state=rnn_state,
                            reset=reset_single,
                            patch_info=patch_info
                        )
                    else:
                         output, rnn_state = self.drc_net.core.forward_single(
                            x_single, rnn_state, reset_single, reset_single
                        )
                else:
                    output, rnn_state = self.drc_net.core.forward_single(
                        x_single, rnn_state, reset_single, reset_single
                    )

                if self.drc_net.record_state: self.drc_net.core.hidden_state.append(torch.concat(rnn_state, dim=1))  

            core_output_list.append(output)

        core_output = torch.cat(core_output_list)
        if self.drc_net.record_state: 
           self.drc_net.core.hidden_state = torch.stack(self.drc_net.core.hidden_state, dim=1)

        core_output = torch.flatten(core_output, 0, 1)

        core_output = torch.cat([x_enc, core_output], dim=1)

        core_output = torch.flatten(core_output, 1)
        final_out = torch.nn.functional.relu(self.drc_net.final_layer(core_output))
        value = self.drc_net.baseline(final_out)
        pri_logits = self.drc_net.policy(final_out)
        pri_logits = pri_logits.view(T*B, self.drc_net.dim_actions, self.drc_net.num_actions)
        pri_probs = torch.nn.functional.softmax(pri_logits.view(-1), dim=0)
        pri = sample(pri_logits, greedy=greedy, dim=-1)
        pri = pri.view(T, B, self.drc_net.dim_actions) 
        pri_env = pri[-1, :, 0] if not self.drc_net.tuple_action else pri[-1]   
        action = pri_env
        return (action, pri_probs, pri_logits.view(-1), rnn_state, value)
    
    def forward_single_patch(self, x, core_state, reset, patch=False, patch_info={}):
        reset = reset.float()

        activ_layers = list(patch_info.keys())

        b, c, h, w = x.shape
        layer_n = 2
        out = core_state[(self.drc_net.core.num_layers - 1) * layer_n] * (1 - reset).view(
            b, 1, 1, 1
        )  # h_cur on last layer

        core_out = []
        new_core_state = []
        for n, cell in enumerate(self.drc_net.core.layers):
            cell_input = torch.concat([x, out], dim=1)
            h_cur = core_state[n * layer_n + 0] * (1 - reset.view(b, 1, 1, 1))
            c_cur = core_state[n * layer_n + 1] * (1 - reset.view(b, 1, 1, 1))
        
            if n in activ_layers and patch is not None:
                if self.debug:
                    print(f"--- Patching Layer {n} ---")

                h_next, c_next = self.forward_cell_patch(
                    convlstm_cell=cell,
                    input=cell_input,
                    h_cur=h_cur,
                    c_cur=c_cur,
                    layer_patch_info=patch_info[n]
                )
            else:
                if self.debug:
                    print(f"--- NOT patching layer {n} ---")
                h_next, c_next, _, _ = cell(
                    cell_input, h_cur, c_cur, None, None, None
                )
            if self.drc_net.core.grad_scale < 1 and h_next.requires_grad:
                h_next.register_hook(lambda grad: grad * self.drc_net.core.grad_scale)
                c_next.register_hook(lambda grad: grad * self.drc_net.core.grad_scale)
            new_core_state.append(h_next)
            new_core_state.append(c_next)
            out = h_next

        core_state = tuple(new_core_state)
        core_out = out.unsqueeze(0)
        return core_out, core_state
    
    def forward_cell_patch(self, convlstm_cell, input, h_cur, c_cur, layer_patch_info):

        combined = torch.cat([input, h_cur], dim=1)  
        if convlstm_cell.pool_inject:
            combined = torch.cat(
                [combined, convlstm_cell.proj_max_mean(h_cur)], dim=1
            )  # concatenate along channel axis

        if convlstm_cell.linear:
            combined_conv = convlstm_cell.main(combined[:, :, 0, 0]).unsqueeze(-1).unsqueeze(-1)
        else:
            combined_conv = convlstm_cell.main(combined)

        cc_i, cc_f, cc_o, cc_g, _ = torch.split(combined_conv, convlstm_cell.embed_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        c_next = f * c_cur + i * g

        for interv_dict in layer_patch_info:
            vec = interv_dict["vec"]
            if self.debug:
                print(f"patching in at cell")
            locs = interv_dict["locs"]
            alpha = interv_dict["alpha"]
            for (y_idx, x_idx) in locs:
                if self.debug:
                    print(f"patching in at loc {y_idx=}, {x_idx=}")
                c_next[0,:,y_idx , x_idx] += alpha*vec

        h_next = o * torch.tanh(c_next)
        return h_next, c_next
    
    
class ProbeIntervResNet:
    """
    Wrapper around ResNet for probe interventions.
    """

    def __init__(self, res_net, debug=False):
        self.res_net = res_net
        self.debug= debug

    def forward_normal(self, env_out, rnn_state):
        return self.res_net(env_out, rnn_state)
    
    def forward_patch(self, env_out: EnvOut, rnn_state: tuple, greedy: bool = True,
                      activ_ticks: Optional[list] = None,
                      patch_info: Optional[dict] = None):
    
        done = env_out.done
        T, B = done.shape
        x = self.res_net.normalize(env_out.real_states.float())
        x = torch.flatten(x, 0, 1)
        x = self.res_net.encoder(x)
        
        for n, layer in enumerate(self.res_net.core):
            x = layer(x)
            if n in patch_info.keys():
                if self.debug:
                    print(f"patching in at layer {n=}")
                for interv_dict in patch_info[n]:
                    vec = interv_dict["vec"]
                    locs = interv_dict["locs"]
                    alpha = interv_dict["alpha"]
                    for (y_idx, x_idx) in locs:
                        if self.debug:
                            print(f"patching in at loc {y_idx=}, {x_idx=}")
                        x[0,:,y_idx, x_idx] += alpha*vec

        core_output = torch.flatten(x, 1)
        final_out = torch.nn.functional.relu(self.res_net.final_layer(core_output))
        
        
        value = self.res_net.baseline(final_out)
        pri_logits = self.res_net.policy(final_out)
        pri_logits = pri_logits.view(T*B, self.res_net.dim_actions, self.res_net.num_actions)
        pri_probs = torch.nn.functional.softmax(pri_logits.view(-1), dim=0)
        pri = sample(pri_logits, greedy=greedy, dim=-1)
        pri = pri.view(T, B, self.res_net.dim_actions) 
        pri_env = pri[-1, :, 0] if not self.res_net.tuple_action else pri[-1]   
        action = pri_env
        return (action, pri_probs, pri_logits.view(-1), (), value)
    
    
paths_3intervs = [
    ([(2,1), (2,0), (3,0)], [(5,1), (5,2), (5,3)], [], [], [], [(2,1)]),
    ([(3,1), (3,0), (4,0)], [(5,1), (5,2), (5,3)], [], [], [], [(3,1)]),
    ([(3,0), (4,0)], [(5,1), (5,2), (5,3)], [], [], [], [(3,0)]),
    ([(2,1), (2,0), (3,0)], [(5,1), (5,2), (5,3)], [], [], [], [(2,1)]),
    ([(2,4), (2,5), (3,5)],  [(5,6), (5,7)], [], [], [(4,7)], [(2,4)]),
    ([(3,4), (3,5), (4,5)], [(5,6), (5,7)], [], [], [(4,7)], [(3,4)]),
    ([(3,3), (3,4), (3,5)], [(5,6), (5,7)], [], [], [(4,7)], [(3,3)]),
    ([(2,1), (3,1), (3,2)], [(4,4), (4,5)], [], [(5,5)], [], [(2,1)]),
    ([(0,3), (0,4), (1,4)], [(2,5), (2,6), (2,7)], [], [], [], [(0,3)]),
    ([(0,2), (0,1), (0,0)], [(4,1), (4,2), (4,3)], [], [], [], [(0,2)]),
    ([(4,3), (5,3)], [(7,4), (7,5)], [], [(7,3)], [], [(4,3)]),
    ([(4,3), (5,3)], [(7,4), (7,5)], [], [(7,3)], [], [(4,3)]),
    ([(3,3), (4,3), (5,3)], [(7,4), (7,5)], [], [(7,3)], [], [(3,3)]),
    ([(1,3), (1,4), (1,5)], [(3,7)], [], [(4,7), (5,7)], [], [(1,3)]),
    ([(1,0), (2,0), (3,0)], [(3,2), (3,3)], [], [(4,3)], [], [(1,0)]),
    ([(1,1), (1,0), (2,0)], [(3,1), (3,2)], [], [(4,2)], [], [(1,1)]),
    ([(1,1), (1,0), (2,0)], [(3,1), (3,2)], [], [(4,2)], [], [(1,1)]),
    ([(2,1), (1,1), (1,2), (1,3)], [(1,6), (1,7)], [], [(2,7)], [], [(2,1)]),
    ([(4,2), (4,3), (4,4)], [], [], [(5,5), (6,5), (7,5)], [], [(4,2)]),
    ([(5,3), (4,3), (3,3), (3,4)], [(4,6), (4,7)], [], [(4,5)], [], [(5,3)]),
    ([(1,1), (0,1), (0,2), (0,3)], [(0,5), (0,6), (0,7)], [], [], [], [(1,1)]),
    ([(1,1), (0,1), (0,2)], [(0,4), (0,5)], [], [(1,5)], [], [(1,1)]),
    ([(3,3), (3,4)], [], [(5,4)], [(4,5), (5,5)], [], [(3,3)]),
    ([(3,4)], [], [(5,4)], [(4,5), (5,5)], [], [(3,4)]),
    ([(3,1), (3,2), (3,3)], [], [], [], [(2,0), (1,0), (0,0)], [(3,1)])
]

paths_2intervs = [
    ([(2,1), (2,0), (3,0)], [(5,1), (5,2)], [], [], [], [(2,1)]),
    ([(3,1), (3,0), (4,0)], [(5,1), (5,2)], [], [], [], [(3,1)]),
    ([(3,0), (4,0)], [(5,1), (5,2)], [], [], [], [(3,0)]),
    ([(2,1), (2,0), (3,0)], [(5,1), (5,2)], [], [], [], [(2,1)]),
    ([(2,4), (2,5), (3,5)],  [(5,6), (5,7)], [], [], [], [(2,4)]),
    ([(3,4), (3,5), (4,5)], [(5,6), (5,7)], [], [], [], [(3,4)]),
    ([(3,3), (3,4), (3,5)], [(5,6), (5,7)], [], [], [], [(3,3)]),
    ([(2,1), (3,1), (3,2)], [(4,4), (4,5)], [], [], [], [(2,1)]),
    ([(0,3), (0,4), (1,4)], [(2,5), (2,6)], [], [], [], [(0,3)]),
    ([(0,2), (0,1), (0,0)], [(4,1), (4,2)], [], [], [], [(0,2)]),
    ([(4,3), (5,3)], [(7,4)], [], [(7,3)], [], [(4,3)]),
    ([(4,3), (5,3)], [(7,4)], [], [(7,3)], [], [(4,3)]),
    ([(3,3), (4,3), (5,3)], [(7,4)], [], [(7,3)], [], [(3,3)]),
    ([(1,3), (1,4), (1,5)], [(3,7)], [], [(4,7)], [], [(1,3)]),
    ([(1,0), (2,0), (3,0)], [(3,2), (3,3)], [], [], [], [(1,0)]),
    ([(1,1), (1,0), (2,0)], [(3,1), (3,2)], [], [], [], [(1,1)]),
    ([(1,1), (1,0), (2,0)], [(3,1), (3,2)], [], [], [], [(1,1)]),
    ([(2,1), (1,1), (1,2), (1,3)], [(1,6), (1,7)], [], [], [], [(2,1)]),
    ([(4,2), (4,3), (4,4)], [], [], [(5,5), (6,5)], [], [(4,2)]),
    ([(5,3), (4,3), (3,3), (3,4)], [(4,6)], [], [(4,5)], [], [(5,3)]),
    ([(1,1), (0,1), (0,2), (0,3)], [(0,5), (0,6)], [], [], [], [(1,1)]),
    ([(1,1), (0,1), (0,2)], [(0,4), (0,5)], [], [], [], [(1,1)]),
    ([(3,3), (3,4)], [], [], [(4,5), (5,5)], [], [(3,3)]),
    ([(3,4)], [], [], [(4,5), (5,5)], [], [(3,4)]),
    ([(3,1), (3,2), (3,3)], [], [], [], [(2,0), (1,0)], [(3,1)])
]

paths_1intervs = [
    ([(2,1), (2,0), (3,0)], [(5,1)], [], [], [], [(2,1)]),
    ([(3,1), (3,0), (4,0)], [(5,1)], [], [], [], [(3,1)]),
    ([(3,0), (4,0)], [(5,1)], [], [], [], [(3,0)]),
    ([(2,1), (2,0), (3,0)], [(5,1)], [], [], [], [(2,1)]),
    ([(2,4), (2,5), (3,5)],  [(5,6)], [], [], [], [(2,4)]),
    ([(3,4), (3,5), (4,5)], [(5,6)], [], [], [], [(3,4)]),
    ([(3,3), (3,4), (3,5)], [(5,6)], [], [], [], [(3,3)]),
    ([(2,1), (3,1), (3,2)], [(4,4)], [], [], [], [(2,1)]),
    ([(0,3), (0,4), (1,4)], [(2,5)], [], [], [], [(0,3)]),
    ([(0,2), (0,1), (0,0)], [(4,1)], [], [], [], [(0,2)]),
    ([(4,3), (5,3)], [], [], [(7,3)], [], [(4,3)]),
    ([(4,3), (5,3)], [], [], [(7,3)], [], [(4,3)]),
    ([(3,3), (4,3), (5,3)], [], [], [(7,3)], [], [(3,3)]),
    ([(1,3), (1,4), (1,5)], [(3,7)], [], [], [], [(1,3)]),
    ([(1,0), (2,0), (3,0)], [(3,2)], [], [], [], [(1,0)]),
    ([(1,1), (1,0), (2,0)], [(3,1)], [], [], [], [(1,1)]),
    ([(1,1), (1,0), (2,0)], [(3,1)], [], [], [], [(1,1)]),
    ([(2,1), (1,1), (1,2), (1,3)], [(1,6)], [], [], [], [(2,1)]),
    ([(4,2), (4,3), (4,4)], [], [], [(5,5)], [], [(4,2)]),
    ([(5,3), (4,3), (3,3), (3,4)], [], [], [(4,5)], [], [(5,3)]),
    ([(1,1), (0,1), (0,2), (0,3)], [(0,5)], [], [], [], [(1,1)]),
    ([(1,1), (0,1), (0,2)], [(0,4)], [], [], [], [(1,1)]),
    ([(3,3), (3,4)], [], [], [(4,5)], [], [(3,3)]),
    ([(3,4)], [], [], [(4,5)], [], [(3,4)]),
    ([(3,1), (3,2), (3,3)], [], [], [], [(2,0)], [(3,1)])
]


paths_0intervs = [
    ([(2,1), (2,0), (3,0)], [], [], [], [], [(2,1)]),
    ([(3,1), (3,0), (4,0)], [], [], [], [], [(3,1)]),
    ([(3,0), (4,0)], [], [], [], [], [(3,0)]),
    ([(2,1), (2,0), (3,0)],[], [], [], [], [(2,1)]),
    ([(2,4), (2,5), (3,5)], [], [], [], [], [(2,4)]),
    ([(3,4), (3,5), (4,5)], [], [], [], [], [(3,4)]),
    ([(3,3), (3,4), (3,5)], [], [], [], [], [(3,3)]),
    ([(2,1), (3,1), (3,2)], [], [], [], [], [(2,1)]),
    ([(0,3), (0,4), (1,4)], [], [], [], [], [(0,3)]),
    ([(0,2), (0,1), (0,0)], [], [], [], [], [(0,2)]),
    ([(4,3), (5,3)], [], [], [], [], [(4,3)]),
    ([(4,3), (5,3)], [], [], [], [], [(4,3)]),
    ([(3,3), (4,3), (5,3)], [], [], [], [], [(3,3)]),
    ([(1,3), (1,4), (1,5)], [], [], [], [], [(1,3)]),
    ([(1,0), (2,0), (3,0)], [], [], [], [], [(1,0)]),
    ([(1,1), (1,0), (2,0)], [], [], [], [], [(1,1)]),
    ([(1,1), (1,0), (2,0)], [], [], [], [], [(1,1)]),
    ([(2,1), (1,1), (1,2), (1,3)], [], [], [], [], [(2,1)]),
    ([(4,2), (4,3), (4,4)], [], [], [], [], [(4,2)]),
    ([(5,3), (4,3), (3,3), (3,4)], [], [], [], [], [(5,3)]),
    ([(1,1), (0,1), (0,2), (0,3)], [], [], [], [], [(1,1)]),
    ([(1,1), (0,1), (0,2)], [], [], [], [], [(1,1)]),
    ([(3,3), (3,4)], [], [], [], [], [(3,3)]),
    ([(3,4)], [], [], [], [], [(3,4)]),
    ([(3,1), (3,2), (3,3)], [], [], [], [], [(3,1)])
]

agent_exp_paths = []
for name, paths in zip([0, 1, 2, 3], [paths_0intervs, paths_1intervs, paths_2intervs, paths_3intervs]):
    olds, new_rs, new_ls, new_ds, new_us, checks = [], [], [], [], [], []
    for old_path, new_right, new_left, new_down, new_up, checkpoint in paths:
        old_path_1 = [(x,7-y) for y,x in old_path]
        new_right_1 = [(x,7-y) for y,x in new_right]
        new_left_1 = [(x,7-y) for y,x in new_left]
        new_down_1 = [(x,7-y) for y,x in new_down]
        new_up_1 = [(x,7-y) for y,x in new_up]
        checkpoint_1 = [(x,7-y) for y,x in checkpoint]

        old_path_2 = [(7-y,7-x) for y,x in old_path]
        new_right_2 = [(7-y,7-x) for y,x in new_right]
        new_left_2 = [(7-y,7-x) for y,x in new_left]
        new_down_2 = [(7-y,7-x) for y,x in new_down]
        new_up_2 = [(7-y,7-x) for y,x in new_up]
        checkpoint_2 = [(7-y,7-x) for y,x in checkpoint]

        old_path_3 = [(7-x,y) for y,x in old_path]
        new_right_3 = [(7-x,y) for y,x in new_right]
        new_left_3 = [(7-x,y) for y,x in new_left]
        new_down_3 = [(7-x,y) for y,x in new_down]
        new_up_3 = [(7-x,y) for y,x in new_up]
        checkpoint_3 = [(7-x,y) for y,x in checkpoint]

        old_path_4 = [(y,7-x) for y,x in old_path]
        new_right_4 = [(y,7-x) for y,x in new_right]
        new_left_4 = [(y,7-x) for y,x in new_left]
        new_down_4 = [(y,7-x) for y,x in new_down]
        new_up_4 = [(y,7-x) for y,x in new_up]
        checkpoint_4 = [(y,7-x) for y,x in checkpoint]

        old_path_5 = [(x,7-y) for y,x in old_path_4]
        new_right_5 = [(x,7-y) for y,x in new_right_4]
        new_left_5 = [(x,7-y) for y,x in new_left_4]
        new_down_5 = [(x,7-y) for y,x in new_down_4]
        new_up_5 = [(x,7-y) for y,x in new_up_4]
        checkpoint_5 = [(x,7-y) for y,x in checkpoint_4]

        old_path_6 = [(7-y,7-x) for y,x in old_path_4]
        new_right_6 = [(7-y,7-x) for y,x in new_right_4]
        new_left_6 = [(7-y,7-x) for y,x in new_left_4]
        new_down_6 = [(7-y,7-x) for y,x in new_down_4]
        new_up_6 = [(7-y,7-x) for y,x in new_up_4]
        checkpoint_6 = [(7-y,7-x)  for y,x in checkpoint_4]

        old_path_7 = [(7-x,y) for y,x in old_path_4]
        new_right_7 = [(7-x,y) for y,x in new_right_4]
        new_left_7 = [(7-x,y) for y,x in new_left_4]
        new_down_7 = [(7-x,y) for y,x in new_down_4]
        new_up_7 = [(7-x,y) for y,x in new_up_4]
        checkpoint_7 = [(7-x,y) for y,x in checkpoint_4]

        olds += [old_path, old_path_1, old_path_2, old_path_3, old_path_4, old_path_5, old_path_6, old_path_7]
        new_rs += [new_right, new_right_1, new_right_2, new_right_3, new_right_4, new_right_5, new_right_6, new_right_7]
        new_ls += [new_left, new_left_1, new_left_2, new_left_3, new_left_4, new_left_5, new_left_6, new_left_7]
        new_ds += [new_down, new_down_1, new_down_2, new_down_3, new_down_4, new_down_5, new_down_6, new_down_7]
        new_us += [new_up, new_up_1, new_up_2, new_up_3, new_up_4, new_up_5, new_up_6, new_up_7]
        checks += [checkpoint, checkpoint_1, checkpoint_2, checkpoint_3, checkpoint_4, checkpoint_5, checkpoint_6, checkpoint_7]
    agent_exp_paths.append([name, (olds, new_rs, new_ls, new_ds, new_us, checks)])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="run intervention experiments in Agent-Shortcut levels")
    parser.add_argument("--model_name", type=str, default="250m", help="name of agent checkpoint on which to run experiments")
    parser.add_argument("--num_seeds", type=int, default=5, help="number of probes to run experiments for")
    parser.add_argument("--num_layers", type=int, default=3, help="number of convlstm layers the agent has")
    parser.add_argument("--num_ticks", type=int, default=3, help="number of internal ticks the agent performs (DRC only)")
    parser.add_argument("--num_episodes", type=int, default=200, help="number of Agent-Shortcut episodes to intervene in")
    parser.add_argument("--noshortrouteinterv", action="store_true", help="flag to not perform short route intervention")
    parser.add_argument('--resnet', action='store_true')
    args = parser.parse_args()

    flags = util.create_setting(args=[], save_flags=False, wrapper_type=1) 
    device = torch.device("cpu")

    for seed in range(args.num_seeds):

        results = []
        dloc_probes = []

        for l in range(args.num_layers):
            probe = ConvProbe(32,5, 1, 0)
            probe.load_state_dict(torch.load(f"./results/convprobe_results/models/agent_onto_after/{args.model_name}_layer{l}_kernel1_wd0.001_seed{seed}.pt", map_location=torch.device('cpu'), weights_only=False))
            dloc_probes.append(probe)
        for l in range(args.num_layers):
            torch.manual_seed((seed*args.num_layers)+l)
            probe = ConvProbe(32,5, 1, 0)
            dloc_probes.append(probe)
        for probe in dloc_probes:
            probe.to(device)

        for layer in range(args.num_layers*2):
            for alpha in [0.25,0.5,1,2,4]:
                alpha_t = alpha
                if layer >= args.num_layers:
                    alpha *= dloc_probes[layer%args.num_layers].conv.weight.norm() / dloc_probes[layer].conv.weight.norm()

                for interv, (olds, new_rs, new_ls, new_ds, new_us, checks) in agent_exp_paths:
                    print(f"========================================= {layer=}, {alpha=}, {interv=}, {seed=}==================================")
                    successes = 0
                    for j in range(args.num_episodes):
                        env = thinker.make(
                                    f"Sokoban-shortcut_clean_{j:04}-v0", 
                                    env_n=1, 
                                    gpu= (True if torch.cuda.is_available() else False),
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
                            if not args.resnet:
                                net = DRCNet(
                                                obs_space=env.observation_space,
                                                action_space=env.action_space,
                                                flags=flags,
                                                record_state=True,
                                                num_layers=args.num_layers,
                                                num_ticks=args.num_ticks
                                                )
                                patch_net = ProbeIntervDRCNet(net, debug=False)
                            else:
                                net = ResNet(
                                    obs_space=env.observation_space,
                                    action_space=env.action_space,
                                    flags=flags,
                                    record_state=True,
                                    num_layers=args.num_layers
                                    )
                                patch_net = ProbeIntervResNet(net, debug=False)
                            ckp_path = "../../checkpoints/sokoban"
                            ckp_path = os.path.join(util.full_path(ckp_path), f"ckp_actor_realstep{args.model_name}.tar")
                            ckp = torch.load(ckp_path, map_location=torch.device('cpu'), weights_only=False)
                            net.load_state_dict(ckp["actor_net_state_dict"], strict=False)
                            net.to(env.device)
                            net.eval()
                            
                     
                        rnn_state = net.initial_state(batch_size=1, device=env.device)
                        state = env.reset()
                        env_out = util.init_env_out(state, flags, dim_actions=1, tuple_action=False)

                        patch_old = True
                        fail = False
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
                        else:
                            raise ValueError("index problem :(")

                        while not done:
                            agent_loc = (state["real_states"][0][4] == 1).to(int).argmax() 
                            agent_x, agent_y = agent_loc % 8, (agent_loc -(agent_loc % 8))//8
                            if (agent_y, agent_x) in new_rs[j] or (agent_y, agent_x) in new_ls[j] or (agent_y, agent_x) in new_us[j] or (agent_y, agent_x) in new_ds[j]:
                                patch_old = False
                            elif (agent_y, agent_x) in checks[j]:
                                fail = True
                            if patch_old:
                                patch_info = {layer % args.num_layers: [{"vec": dloc_probes[layer].conv.weight[0].view(32), "locs": olds[j], "alpha": alpha if not args.noshortrouteinterv else 0},
                                            {"vec": dloc_probes[layer].conv.weight[right_idx].view(32), "locs": new_rs[j], "alpha": alpha},
                                            {"vec": dloc_probes[layer].conv.weight[left_idx].view(32), "locs": new_ls[j], "alpha": alpha},
                                            {"vec": dloc_probes[layer].conv.weight[down_idx].view(32), "locs": new_ds[j], "alpha": alpha},
                                            {"vec": dloc_probes[layer].conv.weight[up_idx].view(32), "locs": new_us[j], "alpha": alpha}] }
                            else:
                                patch_info = {layer % args.num_layers: [{"vec": dloc_probes[layer].conv.weight[0].view(32), "locs": olds[j], "alpha": alpha if not args.noshortrouteinterv else 0}]}
                            patch_action, patch_action_probs, patch_logits, rnn_state, value = patch_net.forward_patch(env_out, rnn_state, activ_ticks=list(range(args.num_ticks)),
                                                                                    patch_info=patch_info)
                            state, reward, done, info = env.step(patch_action)
                            ep_len += 1
                            env_out = util.create_env_out(patch_action, state, reward, done, info, flags)
                        if not fail and ep_len < 115:
                            successes += 1

                    results.append({"layer": layer, "alpha":alpha_t, "intervs": interv, "success_rate": successes / args.num_episodes}) 
    
        if not os.path.exists("./results"):
            os.mkdir("./results")
        if not os.path.exists("./results/interv_results"):
            os.mkdir("./results/interv_results")
        pd.DataFrame(results).to_csv(f"./results/interv_results/agentinterv"+ ("" if not args.noshortrouteinterv else "_noshortrouteinterv") +f"_{args.model_name}_seed{seed}.csv")