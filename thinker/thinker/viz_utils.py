from matplotlib import colors
import matplotlib.pyplot as plt
import numpy as np
from celluloid import Camera
from torch import swapaxes
import os

def plot_mini_sokoban(state, legend=False, unqtar=False, unqbox=False):
    """Plot board state for mini/symbolic sokoban

    Args:
        state (numpy.ndarray): 8x8x7 array representing board
        legend (bool, optional): if True, include legend explaining colours. Defaults to False.
        unqtar (bool, optional): set to True if symbolic sokoban set up in such a way as to track the four target locations as 4 distinct features. Defaults to False.
        unqbox (bool, optional): set to True if symbolic sokoban set up in such a way as to track the boxes as 4 distinct features. Defaults to False.
    """
    if state.shape[0] != state.shape[1]:
        state = state.permute(1,2,0)
    if unqtar and unqbox:
        dim_z = 13
    elif unqtar:
        dim_z = 10
    else:
        dim_z = 7
    mini_board = np.zeros(state.shape[:-1])
    for i in range(1,1+dim_z):
        mini_board[(state[:,:,i-1] == 1)] = i
    mini_board = np.flip(mini_board, axis=0)
    #print(mini_board)
    if unqtar and unqbox:
        cmap = colors.ListedColormap(['black', "white", "blue", "gold", "green","magenta", "salmon", "tomato", "darksalmon", "coral", "cadetblue", "aqua", "powderblue", "steelblue", "orchid"])
        labs = ["wall", "empty", "box1", "box-on-target?", "player","player-on-target?", "target1", "target2", "target3", "target4", "box2", "box3", "box4"]
    elif unqtar:
        cmap = colors.ListedColormap(['black', "white", "aqua", "gold", "green","magenta", "khaki", "grey", "yellow", "purple", "lime"])
        labs = ["wall", "empty", "box", "box-on-target?", "player","player-on-target?", "target1", "target2", "target3", "target4"]
    else:
        cmap = colors.ListedColormap(['black', "white", "aqua", "gold", "green","magenta", "khaki"])
        labs = ["wall", "empty", "box", "box-on-target?", "player","player-on-target?", "target"]
    bounds= [i+0.5 for i in range(1+dim_z)]
    norm = colors.BoundaryNorm(bounds, cmap.N)
    board_img = plt.imshow(mini_board, interpolation='nearest', origin='lower', cmap=cmap, norm=norm)
    board_img.axes.get_yaxis().set_visible(False)
    board_img.axes.get_xaxis().set_visible(False)
    if legend:
        board_cbar = plt.colorbar(board_img, cmap=cmap, norm=norm, boundaries=bounds, ticks=bounds)
        board_cbar.ax.set_yticks(list(range(1,1+dim_z)))
        board_cbar.ax.tick_params(length=0)
        board_cbar_text = board_cbar.ax.set_yticklabels(labs)
    plt.show()

def make_gif_channels(states_list, layer, channels, gif_fps=0.5, gif_file="../viz/interesting_channels", gif_name="test", max_iter=30, max_activ=0, min_activ=0):
    """Make gif of multiple channels across timesteps

    Args:
        states_list (list): list of (agent_states, env_state) tuples, where agent_states is (batch,4,192) tensor of agent hidden states and env_state is (batch,x_dim,y_dim,z_dim) array of board state observations
        layer (int): layer to inspect (must be 0,1,2)
        channels (list): list of channels to inspect
        gif_fps (float, optional): gif speed. Defaults to 0.5.
        gif_file (str, optional): location to store gifs. Defaults to "../viz/interesting_channels".
        gif_name (str, optional): gif file name. Defaults to "test".
        max_iter (int, optional): max number of frames to see. Defaults to 30.
        max_activ (int, optional): cap activations above to this value for comparability if non-zero. Defaults to 0.
        min_activ (int, optional): cap activations below to this value for comparability if non-zero. Defaults to 0.
    """
    if max_activ and min_activ:
        for agent_states, env_state in states_list:
            agent_states[agent_states>max_activ] = max_activ
            agent_states[agent_states<min_activ] = min_activ

    fig, axs = plt.subplots(5,len(channels))
    for tick_idx in range(5):
        for channel_idx in range(len(channels)):
            axs[tick_idx, channel_idx].axis("off")
    camera = Camera(fig)
    i = 0
    for agent_states, env_state in states_list:
        env_state = swapaxes(swapaxes(env_state[-3:].detach().cpu(), 0, 2), 0, 1).numpy()
        for channel_idx, channel in enumerate(channels):
            axs[0,channel_idx].set_title(channel)
            axs[0, channel_idx].imshow(env_state, interpolation="nearest")
            for tick_idx in range(4):
                if max_activ and min_activ:
                    axs[tick_idx+1, channel_idx].imshow(agent_states[tick_idx,layer,channel,:,:].detach(), vmin=min_activ, vmax=max_activ)
                else:
                    axs[tick_idx+1, channel_idx].imshow(agent_states[tick_idx,layer,channel,:,:].detach())
        for layer_idx in range(5):
            for tick_idx in range(len(channels)):
                axs[layer_idx, tick_idx].axis("off")
        plt.pause(0.1)
        camera.snap()
        i += 1
        if i == max_iter:
            break
    animation = camera.animate()
    animation.save(gif_file+"/"+f'{gif_name}.gif', writer='PillowWriter', fps=0.5) 


def make_gif_channel_across_envs(states_list, layer, channel, gif_fps=0.5, gif_name="test", gif_file="../viz/channel_vizs", max_iter=30):
    """Make gif of single channels for multiple environments; states_list needs to be a list where each entry is a tuple of (agent_states, env_state) entries across envs

    Args:
        states_list (list):  list where each entry is a tuple of (agent_states, env_state) entries across envs
        layer (int): layer to inspect (must be 0,1,2)
        channels (int): channels to inspect
        gif_fps (float, optional): gif speed. Defaults to 0.5.
        gif_file (str, optional): location to store gifs. Defaults to "../viz/interesting_channels".
        gif_name (str, optional): gif file name. Defaults to "test".
        max_iter (int, optional): max number of frames to see. Defaults to 30.
    """
    fig, axs = plt.subplots(5,len(states_list[0]))
    fig.suptitle(f"Layer {layer}, channel {channel}")
    for tick_idx in range(5):
        for env_idx in range(len(states_list[0])):
            axs[tick_idx, env_idx].axis("off")
    camera = Camera(fig)
    for i in range(max_iter):
        for env_idx, (agent_states, env_state) in enumerate(states_list[i]):
            env_state = swapaxes(swapaxes(env_state[-3:].detach().cpu(), 0, 2), 0, 1).numpy()
            axs[0, env_idx].imshow(env_state, interpolation="nearest")
            for tick_idx in range(4):
                axs[tick_idx+1, env_idx].imshow(agent_states[tick_idx,layer,channel,:,:].detach())
        plt.pause(0.1)
        camera.snap()
    animation = camera.animate()
    animation.save(gif_file+"/"+f'{gif_name}.gif', writer='PillowWriter', fps=gif_fps)


def create_gif_single_env_multi_channels(agent_env_list, batch, layer, channels, mini=False, max_frames=100, gif_file="./viz", gif_name="test",vmin=-5, vmax=5):
    fig, axs = plt.subplots(5,len(channels))
    for layer_idx in range(5):
        for tick_idx in range(len(channels)):
            axs[layer_idx, tick_idx].axis("off")
    camera = Camera(fig)
    n_frames = 0
    for agent_states, env_states in agent_env_list:
        env_state = env_states[batch]
        if env_state.shape[0] != env_state.shape[1]:
            env_state = env_state.permute(1,2,0)
        if mini:
            mini_board = np.zeros(env_state.shape[:-1])
            for i in range(1,8):
                mini_board[(env_state[:,:,i-1] == 1)] = i
            mini_board = np.flip(mini_board, axis=0)
            cmap = colors.ListedColormap(['black', "lavender", "peru", "gold", "green","magenta", "khaki"])
            bounds=[0.5,1.5,2.5,3.5,4.5,5.5,6.5,7.5]
            norm = colors.BoundaryNorm(bounds, cmap.N)
            board = mini_board
        else:
            board = env_state
        for i in range(len(channels)):
            axs[0][i].imshow(board, interpolation='nearest', origin='lower', cmap=cmap, norm=norm)
        for channel_idx, channel in enumerate(channels):
            axs[0,channel_idx].set_title(channel)
            for tick_idx in range(4):
                axs[tick_idx+1, channel_idx].imshow(agent_states[batch, tick_idx,layer*64+channel,:,:].detach(), vmin=vmin, vmax=vmax)
        plt.pause(0.1)
        camera.snap()
        n_frames += 1
        if n_frames == max_frames:
            break
    animation = camera.animate()
    if not os.path.exists(gif_file):
        os.makedirs(gif_file)
    animation.save(gif_file+"/"+f'{gif_name}.gif', writer='PillowWriter', fps=0.5) 


def create_gif_multi_env_single_channel(agent_env_list, envs, layer, channel, mini=False, max_frames=100, gif_file="./viz", gif_name="test", vmin=-5, vmax=5):
    fig, axs = plt.subplots(5,len(envs))
    for layer_idx in range(5):
        for tick_idx in range(len(envs)):
            axs[layer_idx, tick_idx].axis("off")
    camera = Camera(fig)
    n_frames = 0
    for agent_states, env_states in agent_env_list:
        for env_idx, env in enumerate(envs):
            env_state = env_states[env]
            if env_state.shape[0] != env_state.shape[1]:
                env_state = env_state.permute(1,2,0)
            if mini:
                mini_board = np.zeros(env_state.shape[:-1])
                for i in range(1,8):
                    mini_board[(env_state[:,:,i-1] == 1)] = i
                mini_board = np.flip(mini_board, axis=0)
                cmap = colors.ListedColormap(['black', "lavender", "peru", "gold", "green","magenta", "khaki"])
                bounds=[0.5,1.5,2.5,3.5,4.5,5.5,6.5,7.5]
                norm = colors.BoundaryNorm(bounds, cmap.N)
                board = mini_board
            else:
                board = env_state
            axs[0][env_idx].imshow(board, interpolation='nearest', origin='lower', cmap=cmap, norm=norm)
            axs[0][env_idx].set_title(env)
            for tick_idx in range(4):
                axs[tick_idx+1, env_idx].imshow(agent_states[env, tick_idx,layer*64+channel,:,:].detach(), vmin=vmin, vmax=vmax)
        plt.pause(0.1)
        camera.snap()
        n_frames += 1
        if n_frames == max_frames:
            break
    animation = camera.animate()
    if not os.path.exists(gif_file):
        os.makedirs(gif_file)
    animation.save(gif_file+"/"+f'{gif_name}.gif', writer='PillowWriter', fps=0.5)    