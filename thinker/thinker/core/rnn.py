import torch
from torch import nn
from torch import Tensor
from torch.nn import Module
import torch.nn.functional as F
import math


class ConvAttnLSTMCell(nn.Module):
    def __init__(
        self,
        input_dims,
        embed_dim,
        kernel_size=3,
        num_heads=8,
        mem_n=8,
        attn=True,
        attn_mask_b=3,
        pool_inject=False,
    ):
        super(ConvAttnLSTMCell, self).__init__()
        c, h, w = input_dims

        self.input_dims = input_dims
        self.linear = h == w == 1
        self.embed_dim = embed_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2

        self.gates = torch.tensor([])

        in_channels = c + self.embed_dim

        #pool_inject = False # ADDED
        if pool_inject:
            in_channels += self.embed_dim

        if self.linear:
            self.main = nn.Linear(in_channels, 5 * self.embed_dim)
        else:
            self.main = nn.Conv2d(
                in_channels=in_channels,
                out_channels=5 * self.embed_dim,
                kernel_size=self.kernel_size,
                padding=self.padding,
            )

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.mem_n = mem_n
        self.head_dim = embed_dim // num_heads
        self.attn = attn
        self.attn_mask_b = attn_mask_b
        #pool_inject = False # ADDED
        self.pool_inject = pool_inject

        if self.attn:
            if self.linear:
                self.proj = nn.Linear(c, self.embed_dim * 3)
                self.out = nn.Linear(self.embed_dim, self.embed_dim)
            else:
                self.proj = torch.nn.Conv2d(
                    in_channels=c,
                    out_channels=self.embed_dim * 3,
                    kernel_size=kernel_size,
                    padding="same",
                )
                self.out = torch.nn.Conv2d(
                    in_channels=self.embed_dim,
                    out_channels=self.embed_dim,
                    kernel_size=kernel_size,
                    padding="same",
                )

            self.norm = nn.modules.normalization.LayerNorm((embed_dim, h, w), eps=1e-5)
            self.pos_w = torch.nn.Parameter(torch.zeros(self.mem_n, h * w * embed_dim))
            self.pos_b = torch.nn.Parameter(torch.zeros(self.mem_n, self.num_heads))
            torch.nn.init.xavier_uniform_(self.pos_w)
            torch.nn.init.uniform_(self.pos_b, -0.1, 0.1)

        if self.pool_inject:
            self.proj = torch.nn.Conv2d(embed_dim, embed_dim, (2, 1), groups=embed_dim)

    def proj_max_mean(self, out):
        out_mean = torch.mean(out, dim=(-1, -2), keepdim=True)
        out_max = torch.max(
            torch.max(out, dim=-1, keepdim=True)[0], dim=-2, keepdim=True
        )[0]
        proj_in = torch.cat([out_mean, out_max], dim=-2)
        out_sum = self.proj(proj_in).broadcast_to(out.shape)
        return out_sum

    def forward(self, input, h_cur, c_cur, concat_k, concat_v, attn_mask):
        """
        Args:
          input (tensor): network input; shape (B, C, H, W)
          h_cur (tensor): previous output; shape (B, embed_dim, H, W)
          c_cur (tensor): previous lstm state; shape (B, embed_dim, H, W)
          concat_k (tensor): previous attn k; shape (B, num_head, mem_n, total_dim)
          concat_v (tensor): previous attn v; shape (B, num_head, mem_n, total_dim)
          attn_mask (tensor): attn mask; shape (B * num_head, 1, mem_n)
        Return:
          h_next (tensor): current output; shape (B, embed_dim, H, W)
          c_next (tensor): current lstm state; shape (B, embed_dim, H, W)
          concat_k (tensor): current attn k; shape (B, num_head, mem_n, total_dim)
          concat_v (tensor): current attn v; shape (B, num_head, mem_n, total_dim)
        """

        B = input.shape[0]
        combined = torch.cat([input, h_cur], dim=1)  # concatenate along channel axis
        if self.pool_inject:
            combined = torch.cat(
                [combined, self.proj_max_mean(h_cur)], dim=1
            )  # concatenate along channel axis

        if self.linear:
            combined_conv = self.main(combined[:, :, 0, 0]).unsqueeze(-1).unsqueeze(-1)
        else:
            combined_conv = self.main(combined)
        cc_i, cc_f, cc_o, cc_g, cc_a = torch.split(combined_conv, self.embed_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g

        if self.attn:
            a = torch.sigmoid(cc_a)
            attn_out, concat_k, concat_v = self.attn_output(
                input, attn_mask, concat_k, concat_v
            )
            c_next = c_next + a * torch.tanh(attn_out)
            self.a = a
        else:
            concat_k, concat_v = None, None

        h_next = o * torch.tanh(c_next)
        self.gates = torch.cat([i, f, o, g], dim=1)

        return h_next, c_next, concat_k, concat_v

    def attn_output(self, input, attn_mask, concat_k, concat_v):
        b, c, h, w = input.shape
        tot_head_dim = h * w * self.embed_dim // self.num_heads

        if self.linear:
            kqv = self.proj(input[:, :, 0, 0]).unsqueeze(-1).unsqueeze(-1)
        else:
            kqv = self.proj(input)

        kqv_reshape = kqv.view(b * self.num_heads, self.head_dim * 3, h * w)
        k, q, v = torch.split(kqv_reshape, self.head_dim, dim=1)
        k, q, v = [
            torch.flatten(x.unsqueeze(0), start_dim=2).transpose(0, 1)
            for x in [k, q, v]
        ]

        q_scaled = q / math.sqrt(q.shape[2])
        k_pre = concat_k.view(b * self.num_heads, -1, tot_head_dim)
        k = torch.cat([k_pre[:, 1:], k], axis=1)

        pos_w = (
            self.pos_w.unsqueeze(1)
            .broadcast_to(self.mem_n, b, -1)
            .contiguous()
            .view(self.mem_n, b * self.num_heads, -1)
            .transpose(0, 1)
        )
        pos_b = (
            self.pos_b.unsqueeze(1)
            .broadcast_to(self.mem_n, b, -1)
            .contiguous()
            .view(self.mem_n, b * self.num_heads)
            .transpose(0, 1)
        )

        k = k + pos_w

        v_pre = concat_v.view(b * self.num_heads, -1, tot_head_dim)
        v = torch.cat([v_pre[:, 1:], v], axis=1)

        new_attn_mask = torch.zeros_like(attn_mask, dtype=q.dtype)
        new_attn_mask.masked_fill_(attn_mask, float("-inf"))
        attn_mask = new_attn_mask
        attn_mask[:, :, -1] = self.attn_mask_b
        self.attn_mask = attn_mask
        attn_weights = torch.baddbmm(attn_mask, q_scaled, k.transpose(-2, -1))
        attn_weights = attn_weights + pos_b.unsqueeze(1)
        attn_weights = torch.softmax(attn_weights, dim=-1)
        self.attn_weights = attn_weights

        attn_output = torch.bmm(attn_weights, v)
        attn_output = attn_output.transpose(0, 1).view(b, self.embed_dim, h, w)

        if self.linear:
            out = self.out(attn_output[:, :, 0, 0]).unsqueeze(-1).unsqueeze(-1)
        else:
            out = self.out(attn_output)
        out = out + input[:, : self.embed_dim]
        out = self.norm(out)

        ret_k = k.view(b, self.num_heads, self.mem_n, tot_head_dim)
        ret_v = v.view(b, self.num_heads, self.mem_n, tot_head_dim)

        return out, ret_k, ret_v


class ConvAttnLSTM(nn.Module):
    def __init__(
        self,        
        input_dim,
        hidden_dim,        
        num_layers,        
        attn,
        h=1,
        w=1,
        kernel_size=1,
        mem_n=None,
        num_heads=8,
        attn_mask_b=5,
        tran_t=1,
        grad_scale=1,
        pool_inject=False,
    ):
        super(ConvAttnLSTM, self).__init__()

        self.h = h
        self.w = w
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.mem_n = mem_n
        self.grad_scale = grad_scale
        self.attn = attn
        self.tran_t = tran_t        
        self.tot_head_dim = h * w * hidden_dim // num_heads

        layers = []

        for i in range(0, self.num_layers):
            layers.append(
                ConvAttnLSTMCell(
                    input_dims=(input_dim + hidden_dim, self.h, self.w),
                    embed_dim=self.hidden_dim,
                    kernel_size=self.kernel_size,
                    num_heads=num_heads,
                    mem_n=mem_n,
                    attn=attn,
                    attn_mask_b=attn_mask_b,
                    pool_inject=pool_inject,
                )
            )

        self.layers = nn.ModuleList(layers)

    def initial_state(self, bsz, device=None):
        core_state = ()
        for _ in range(self.num_layers):
            core_state = core_state + (
                torch.zeros(bsz, self.hidden_dim, self.h, self.w, device=device),
                torch.zeros(bsz, self.hidden_dim, self.h, self.w, device=device),
            )
            if self.attn:
                core_state = core_state + (
                    torch.zeros(
                        bsz,
                        self.num_heads,
                        self.mem_n,
                        self.tot_head_dim,
                        device=device,
                    ),
                    torch.zeros(
                        bsz,
                        self.num_heads,
                        self.mem_n,
                        self.tot_head_dim,
                        device=device,
                    ),
                )
        if self.attn:
            core_state = core_state + (
                torch.ones(bsz, self.mem_n, device=device).bool(),
            )
        return core_state
    
    def forward(self, x, done, core_state, record_state=False, record_output=False, record_gates=False):
        assert len(x.shape) == 5
        core_output_list = []
        reset = done.float()
        if record_state: 
            self.hidden_state = []
            self.hidden_state.append(torch.concat(core_state, dim=1)) 
        if record_output:
            self.output = []
        if record_gates:
            self.gates = [] 
        for n, (x_single, reset_single) in enumerate(
            zip(x.unbind(), reset.unbind())
        ):
            gates = []
            for t in range(self.tran_t):
                if t > 0:
                    reset_single = torch.zeros_like(reset_single)
                reset_single = reset_single.view(-1)
                output, core_state = self.forward_single(
                    x_single, core_state, reset_single, reset_single
                )  # output shape: 1, B, core_output_size        
                if record_state: self.hidden_state.append(torch.concat(core_state, dim=1))
                if record_gates: self.gates.append(torch.cat([cell.gates for cell in self.layers], dim=0))
                if record_output: self.output.append(output)              
            core_output_list.append(output)
        core_output = torch.cat(core_output_list)
        if record_state: 
            self.hidden_state = torch.stack(self.hidden_state, dim=1)
        if record_output:
            self.output = torch.cat(self.output, dim=0)
        if record_gates:
            self.gates = torch.cat(self.gates, dim=1)
        return core_output, core_state

    def forward_single(self, x, core_state, reset, reset_attn):
        reset = reset.float()
        if reset_attn is None:
            reset_attn = reset.float()
        else:
            reset_attn = reset_attn.float()

        b, c, h, w = x.shape
        layer_n = 4 if self.attn else 2
        out = core_state[(self.num_layers - 1) * layer_n] * (1 - reset).view(
            b, 1, 1, 1
        )  # h_cur on last layer
        
        if self.attn:
            src_mask = core_state[-1]
            src_mask[reset_attn.bool(), :] = True
            src_mask[:, :-1] = src_mask[:, 1:].clone().detach()
            src_mask[:, -1] = False
            new_src_mask = src_mask
            src_mask_reshape = (
                src_mask.view(b, 1, 1, -1)
                .broadcast_to(b, self.num_heads, 1, -1)
                .contiguous()
                .view(b * self.num_heads, 1, -1)
            )
        else:
            src_mask_reshape = None

        core_out = []
        new_core_state = []
        for n, cell in enumerate(self.layers):
            cell_input = torch.concat([x, out], dim=1)
            h_cur = core_state[n * layer_n + 0] * (1 - reset.view(b, 1, 1, 1))
            c_cur = core_state[n * layer_n + 1] * (1 - reset.view(b, 1, 1, 1))
            concat_k_cur = core_state[n * layer_n + 2] if self.attn else None
            concat_v_cur = core_state[n * layer_n + 3] if self.attn else None

            h_next, c_next, concat_k, concat_v = cell(
                cell_input, h_cur, c_cur, concat_k_cur, concat_v_cur, src_mask_reshape
            )
            if self.grad_scale < 1 and h_next.requires_grad:
                h_next.register_hook(lambda grad: grad * self.grad_scale)
                c_next.register_hook(lambda grad: grad * self.grad_scale)
            if self.grad_scale < 1 and self.attn and concat_k.requires_grad:
                concat_k.register_hook(lambda grad: grad * self.grad_scale)
                concat_v.register_hook(lambda grad: grad * self.grad_scale)

            new_core_state.append(h_next)
            new_core_state.append(c_next)
            if self.attn:
                new_core_state.append(concat_k)
                new_core_state.append(concat_v)
            out = h_next

        core_state = tuple(new_core_state)
        if self.attn:
            core_state = core_state + (new_src_mask,)

        core_out = out.unsqueeze(0)
        return core_out, core_state
    
class LSTMReset(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(LSTMReset, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers)
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

    def initial_state(self, bsz, device=None):
        return (torch.zeros(bsz, self.num_layers, self.hidden_dim, device=device),
                torch.zeros(bsz, self.num_layers, self.hidden_dim, device=device)
                )

    def forward(self, input, reset, state):
        # input shape: (seq_len, batch, input_size)
        # reset shape: (seq_len, batch), dtype=torch.bool
        
        seq_len = input.shape[0]
        reset = reset.float()

        state = tuple(s.transpose(0,1).contiguous() for s in state)

        outputs = []
        for t in range(seq_len):
            # Process one timestep
            input_t = input[t].unsqueeze(0)
            state_reset = tuple(s * (1 - reset[t]).unsqueeze(-1) for s in state)
            output_t, state = self.lstm(input_t, state_reset)
            outputs.append(output_t)

        state = tuple(s.transpose(0,1) for s in state)

        outputs = torch.cat(outputs, dim=0)
        return outputs, state    
    

class ResBlock(nn.Module):

    def __init__(self, in_dims: int, out_dims: int, kernel_size: int, stride: int, record_state: bool = False):
        super(ResBlock, self).__init__()

        self.conv1 = nn.Conv2d(out_dims, out_dims, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
        self.bn1 = nn.BatchNorm2d(out_dims)
        self.conv2 = nn.Conv2d(out_dims, out_dims, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
        self.bn2 = nn.BatchNorm2d(out_dims)
        
        self.record_state = record_state

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = F.relu(out+x)
        if self.record_state:
            self.hidden_state = out 
        return out
