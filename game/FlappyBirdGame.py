'''
This program implements the game environment of Flappy Bird.
'''

import numpy as np
import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image

import gym
from gym.wrappers import Monitor
import gym_ple
from gym_ple import PLEEnv
from utils import *

import warnings
warnings.filterwarnings('ignore')


class FlappyBirdNormal(gym.Wrapper):
    ''' Game environment for SARSA and Q-Learning. '''
    
    def __init__(self, env, rounding = None):
        '''
        Initializes the environment.
        
        Args:
            env (PLEEnv): A Pygame environment.
            rounding (int): The level of discretization.
        '''
        super().__init__(env)
        self.rounding = rounding
    
    def save_output(self, outdir = None):
        '''
        Saves videos of the game.
        
        Args:
            outdir (str): Output directory.
        '''
        if outdir:
            self.env = Monitor(self.env, directory = outdir, force = True)
        
    def step(self, action):
        '''
        Lets the agent take an action and observe the next state and reward.
        
        Args:
            action (int): 0 or 1.
        
        Returns:
            tuple: state, reward, terminal.
        '''
        _, reward, terminal, _ = self.env.step(action)
        state = self.getGameState()
        if not terminal: reward += 0.5
        else: reward = -1000
        if reward >= 1: reward = 5
        return state, reward, terminal, {}

    def getGameState(self):
        '''
        Returns the current game state.
        
        Returns:
            str: A string representing the game state.
        '''
        gameState = self.env.game_state.getGameState()
        hor_dist_to_next_pipe = gameState['next_pipe_dist_to_player']
        ver_dist_to_next_pipe = gameState['next_pipe_bottom_y'] - gameState['player_y']
        if self.rounding:
            hor_dist_to_next_pipe = discretize(hor_dist_to_next_pipe, self.rounding)
            ver_dist_to_next_pipe = discretize(ver_dist_to_next_pipe, self.rounding)
            
        state = []
        state.append('player_vel' + ' ' + str(gameState['player_vel']))
        state.append('hor_dist_to_next_pipe' + ' ' + str(hor_dist_to_next_pipe))
        state.append('ver_dist_to_next_pipe' + ' ' + str(ver_dist_to_next_pipe))
        return ' '.join(state)
        

class FlappyBirdLR(gym.Wrapper):
    ''' Game environment for Function Approximation with Linear Regression. '''
    
    def __init__(self, env):
        '''
        Initializes the environment.
        
        Args:
            env (PLEEnv): A Pygame environment.
        '''
        super().__init__(env)
    
    def save_output(self, outdir = None):
        '''
        Saves videos of the game.
        
        Args:
            outdir (str): Output directory.
        '''
        if outdir:
            self.env = Monitor(self.env, directory = outdir, force = True)
        
    def step(self, action):
        '''
        Lets the agent take an action and observe the next state and reward.
        
        Args:
            action (int): 0 or 1.
        
        Returns:
            tuple: state, reward, terminal.
        '''
        _, reward, terminal, _ = self.env.step(action)
        state = self.getGameState()
        if not terminal: reward += 0.5
        else: reward -= 1000
        if reward >= 1: reward = 5
        return state, reward, terminal, {}

    def getGameState(self):
        '''
        Returns the current game state.
        
        Returns:
            dict: A dictionary representing the game state.
        '''
        return self.env.game_state.getGameState()
        
        
class FlappyBirdDNN(gym.Wrapper):
    ''' Game environment for Function Approximation with Feed Forward Neural Networks. '''
    
    def __init__(self, env):
        '''
        Initializes the environment.
        
        Args:
            env (PLEEnv): A Pygame environment.
        '''
        super().__init__(env)
    
    def save_output(self, outdir = None):
        '''
        Saves videos of the game.
        
        Args:
            outdir (str): Output directory.
        '''
        if outdir:
            self.env = Monitor(self.env, directory = outdir, force = True)
        
    def step(self, action):
        '''
        Lets the agent take an action and observe the next state and reward.
        
        Args:
            action (int): 0 or 1.
        
        Returns:
            tuple: state, reward, terminal.
        '''
        _, reward, terminal, _ = self.env.step(action)
        state = self.getGameState()
        if not terminal: reward += 0.5
        else: reward = -1000
        if reward >= 1: reward = 5
        return state, reward, terminal, {}

    def getGameState(self):
        '''
        Returns the current game state.
        
        Returns:
            list: A list representing the game state.
        '''
        gameState = self.env.game_state.getGameState()
        hor_dist_to_next_pipe = gameState['next_pipe_dist_to_player']
        ver_dist_to_next_pipe = gameState['next_pipe_bottom_y'] - gameState['player_y']
        state = []
        state.append(gameState['player_vel'])
        state.append(hor_dist_to_next_pipe)
        state.append(ver_dist_to_next_pipe)
        return state
            

class FlappyBirdCNN(gym.Wrapper):
    ''' Game environment for Function Approximation with Convolutional Neural Networks. '''
    
    def __init__(self, env):
        '''
        Initializes the environment.
        
        Args:
            env (PLEEnv): A Pygame environment.
        '''
        super().__init__(env)
        self.transform = Transform()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def save_output(self, outdir = None):
        '''
        Saves videos of the game.
        
        Args:
            outdir (str): Output directory.
        '''
        if outdir:
            self.env = Monitor(self.env, directory = outdir, force = True)
        
    def step(self, action):
        '''
        Lets the agent take an action and observe the next state and reward.
        
        Args:
            action (int): 0 or 1.
        
        Returns:
            tuple: state, reward, terminal.
        '''
        state, reward, terminal, _ = self.env.step(action)
        state = self.transform.process(state)
        if not terminal: reward += 0.5
        else: reward = -1000
        if reward >= 1: reward = 5
        return state, reward, terminal, {}

    def getGameState(self):
        '''
        Returns the current game state.
        
        Returns:
            list: A list representing the game state.
        '''
        gameState = self.env.game_state.getGameState()
        return list(gameState.values())


class Transform(object):
    ''' A class that preprocesses the images of the game screen. '''
    
    def __init__(self):
        ''' Initializes the class. '''
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.transform = transforms.Compose([
              transforms.Lambda(lambda x: x[:404]),
              transforms.ToPILImage(),
              transforms.Grayscale(),
              transforms.Resize((80, 80)),
              transforms.Lambda(lambda x:
                    cv2.threshold(np.array(x), 128, 255, cv2.THRESH_BINARY)[1]),
              transforms.ToTensor(),
              transforms.Lambda(lambda x: x.unsqueeze(0).to(self.device)),
        ])

    def process(self, img):
        '''
        Transforms the input image.
        
        Args:
            img (ndarray): Imput image.
            
        Returns:
            Tensor: A transformed image.
        '''
        return self.transform(img)
