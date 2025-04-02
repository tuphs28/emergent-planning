import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import pygame
import sys
import gym_pilleater
import gym
import os
import thinker
import torch

os.environ["SDL_VIDEODRIVER"] = "x11"
os.environ["DISPLAY"] = ":0"  # Adjust if necessary

flags = thinker.util.create_setting(args=[], save_flags=False, wrapper_type=1) 
env = thinker.make(
            f"gym_pilleater/PillEater-v0", 
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

state = env.reset() 
# Initialize pygame for keyboard input
pygame.init()
pygame.display.set_mode((100, 100))  # Create a small window to capture events


# Create action mappings for WASD keys
KEY_MAPPING = {
    pygame.K_w: 2,  # Up
    pygame.K_s: 1,  # Down
    pygame.K_a: 3,  # Left
    pygame.K_d: 4   # Right
}

# Function to render the board
def render_board(state):
    """Render the game board using matplotlib."""
    mini_board = np.zeros((13, 13))
    for i in range(1, 15):
        mini_board[state["real_states"][0][i-1,:, :] == 1] = i

    cmap = colors.ListedColormap([
        "black", # wall
        "silver", # food
        "red", # g
        "aqua", # ge
        "skyblue", # gee
        "white", # not food
        "tomato", # g
        "cyan", # ge
        "lavender", # gee
        "yellow", # pill
        "gold", # g 
        "peru", # ge
        "linen", # gee
        "darkgreen", # pillbro
    ])
    bounds = [i+0.5 for i in range(15)]
    norm = colors.BoundaryNorm(bounds, cmap.N)

    plt.imshow(mini_board, interpolation='nearest', origin='lower', cmap=cmap, norm=norm)
    plt.xticks([])
    plt.yticks([])
    plt.draw()
    # plt.pause(0.001)  # Removed this line

# Initialize matplotlib figure
plt.ion()  # Interactive mode on
fig, ax = plt.subplots()

# Render initial state and display the plot window
render_board(state)
plt.show(block=False)  # Use block=False to prevent blocking the script

# Function to handle pygame event loop with frame control
def game_loop():
    global done, state

    # Frame control
    clock = pygame.time.Clock()
    print("Game loop started")

    while True:
        for event in pygame.event.get():
            #print(event)  # Debugging: print events
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key in KEY_MAPPING:
                    # Get action from the key pressed
                    action = torch.tensor([KEY_MAPPING[event.key]])
                    # Step the environment
                    state, reward, done, _ = env.step(action)
                    plt.clf()  # Clear the previous plot                
                    if done:
                        # If the game ends, reset the environment
                        print("Game Over! Resetting...")
                        #state = env.reset()
                    else:
                        render_board(state)
                        

        # Allow Matplotlib to process events and update the plot
        plt.pause(0.001)

        # Control the frame rate (increase for smoother updates)
        clock.tick(30)
        # time.sleep(0.01)  # Optional: prevent 100% CPU usage if needed

# Run the game loop
game_loop()
