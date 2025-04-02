import gym
from gym.spaces.discrete import Discrete
from gym.spaces import Box
from .csokoban import cSokoban
import numpy as np
import torch
import pkg_resources
import os 

class SokobanEnv(gym.Env):
    def __init__(self, difficulty='unfiltered', small=True, dan_num=0, seed=0, mini=True, mini_unqtar=False, mini_unqbox=False):
        if difficulty == 'unfiltered': 
            level_num = 900000                      
            path = '/'.join(('boxoban-levels', difficulty, 'train'))
        elif difficulty == 'easy': 
            level_num = 17000                      
            path = '/'.join(('boxoban-levels', 'unfiltered', 'easy'))
        elif difficulty == 'test':     
            level_num = 1000                  
            path = '/'.join(('boxoban-levels', 'unfiltered', 'test'))
        elif difficulty == "probing":
            level_num = 5000
            path = '/'.join(('boxoban-levels', 'unfiltered', 'probing'))
        elif difficulty == "valid":
            level_num = 100000
            path = '/'.join(('boxoban-levels', 'unfiltered', 'valid'))
        elif difficulty == 'medium':            
            level_num = 50000           
            path = '/'.join(('boxoban-levels', difficulty, 'valid'))
        elif difficulty == 'hard': 
            level_num = 3332
            print("hard diff.")         
            path = '/'.join(('boxoban-levels', difficulty))
        elif difficulty[:3] == 'exp': 
            level_num = 1
            _, exp_type, exp_baseline, exp_id = difficulty.split("_")
            path = '/'.join(('boxoban-levels', 'experiments', exp_type, exp_id, exp_baseline))
        else:
            raise Exception(f"difficulty {difficulty} not accepted.")

        level_dir = pkg_resources.resource_filename(__name__, path)
        img_dir = pkg_resources.resource_filename(__name__, 'surface')

        self.mini = mini
        self.mini_unqtar = mini_unqtar
        self.mini_unqbox = mini_unqbox
        
        self.sokoban = cSokoban(small=small, 
                                level_dir=level_dir.encode('UTF-8'), 
                                img_dir=img_dir.encode('UTF-8'), 
                                level_num=level_num, 
                                dan_num=dan_num,
                                seed=seed,
                                mini=mini,
                                mini_unqtar=mini_unqtar,
                                mini_unqbox=mini_unqbox)
        self.action_space = Discrete(5)
        if mini:
            if mini_unqbox and mini_unqtar:
                obs_z = 13
            elif mini_unqtar and not mini_unqbox:
                obs_z = 10
            else:
                obs_z = 7
            self.observation_space = Box(low=0, high=1, shape=(self.sokoban.obs_x-2, self.sokoban.obs_y-2, obs_z), dtype=np.uint8)
        else:
            self.observation_space = Box(low=0, high=255, shape=(self.sokoban.obs_x, self.sokoban.obs_y, 3), dtype=np.uint8)

        # self.sokoban.reset()

    def step(self, action):
        obs, reward, done, info = self.sokoban.step(action)
        reward = round(reward, 2)
        return obs, reward, done, info

    def reset(self, room_id=None):
        if room_id is None:
            return self.sokoban.reset()
        else:
            return self.sokoban.reset_level(room_id)   
        
    def clone_state(self):
        return self.sokoban.clone_state()

    def restore_state(self, state):
        return self.sokoban.restore_state(state)    

    def seed(self, seed): 
        self.sokoban.seed(seed)    

    @property
    def step_n(self):
        return self.sokoban.step_n

    @step_n.setter
    def step_n(self, step_n):
        self.sokoban.step_n = step_n 

