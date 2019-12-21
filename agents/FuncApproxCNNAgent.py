'''
This program implements a Function Approximation Agent with a Convolutional Neural Network.
'''

from __future__ import print_function

import os, sys
sys.path.append('../game')
sys.path.append('../utils')

from collections import defaultdict, deque
import json
import random
import numpy as np

import gym
from TemplateAgent import FlappyBirdAgent
from FlappyBirdGame import FlappyBirdCNN

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.init as init
import torchvision.transforms as transforms
from PIL import Image

import warnings
warnings.filterwarnings('ignore')


def conv2d_size_out(size, kernel_size = 5, stride = 2):
    '''
    Calculates the shape of the output of a convolutional layer.
    
    Args:
        size (int): The size of the input image.
        kernel_size (int): The size of the kernel.
        stride (int): The stride value.
        
    Returns:
        int: The size of the output feature map.
    '''
    return (size - (kernel_size - 1) - 1) // stride  + 1

class Net(nn.Module):
    ''' Convolutional Neural Network. '''
    
    CONVW = conv2d_size_out(conv2d_size_out(80))
    CONVH = conv2d_size_out(conv2d_size_out(80))
    LINEAR_INPUT_SIZE_FC1 = CONVW * CONVH * 32
    
    def __init__(self):
        ''' Initializes the network architecture. '''
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size = 5, stride = 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size = 5, stride = 2)
        self.fc1 = nn.Linear(self.LINEAR_INPUT_SIZE_FC1, 2)
        
    def forward(self, x):
        '''
        Performs a forward pass through the network.
        
        Args:
            x (Tensor): An input image.
            
        Returns:
            Tensor: An output vector.
        '''
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x.squeeze(0)


class ExperienceReplay(object):
    ''' Experience Replay technique, which samples a minibatch of past observations. '''
    
    def __init__(self, size):
        '''
        Initializes the memory.
        
        Args:
            size (int): The maximum size of the memory.
        '''
        self.memory = deque(maxlen = size)

    def __len__(self):
        '''
        Returns the number of observations in the memory.
        
        Returns:
            int: The number of observations in the memory.
        '''
        return len(self.memory)

    def memorize(self, observation):
        '''
        Adds a new observation to the memory.
        
        Args:
            onservation (tuple): A new observation.
        '''
        self.memory.append(observation)
        
    def getBatch(self, batch_size):
        '''
        Samples a minibatch of observations from the memory.
        
        Args:
            batch_size (int): The size of a minibatch.
            
        Returns:
            zip: A minibatch of observations.
        '''
        if len(self.memory) < batch_size:
            return None
        batch = random.sample(self.memory, batch_size)
        return zip(*batch)


class HParams():
    ''' A class storing hyperparameters of the model. '''
    
    def __init__(self, lr = 0.1, resume = False, seed = 0, 
                 batch_size = 32, start_epoch = 0, epoch = 10000, 
                 decay = 1e-4, num_experience = 50000):
        '''
        Initializes the parameters.
        
        Args:
            lr (float): The learning rate.
            resume (bool): Whether to resume from checkpoints.
            seed (int): Random seed for PyTorch.
            batch_size (int): The size of a batch.
            start_epoch (int): The starting epoch.
            epoch (int): The number of epochs.
            decay (float); Weight decay.
            num_experience (int): The size of the Experience Replay memory.
        '''
        self.lr = lr
        self.resume = resume
        self.seed = seed
        self.batch_size = batch_size
        self.start_epoch = start_epoch
        self.epoch = epoch
        self.decay = decay
        self.num_experience = num_experience


class FuncApproxCNNAgent(FlappyBirdAgent):
    ''' Function Approximation Agent with a Convolutional Neural Network. '''
    
    def __init__(self, actions, probFlap = 0.1):
        '''
        Initializes the agent.
        
        Args:
            actions (list): Possible action values.
            probFlap (float): The probability of flapping when choosing
                              the next action randomly.
        '''
        super().__init__(actions)
        self.probFlap = probFlap
        self.env = FlappyBirdCNN(gym.make('FlappyBird-v0'))
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.net = Net().to(self.device)
        self.criterion = torch.nn.MSELoss()
        
        # 0 corresponds to [1, 0], 1 corresponds to [0, 1].
        self.actionEncoding = torch.eye(2, device = self.device).unsqueeze(1)

    def act(self, state):
        '''
        Returns the next action for the current state.
        
        Args:
            state (str): The current state.
            
        Returns:
            int: 0 or 1.
        '''
        def randomAct():
            if random.random() < self.probFlap:
                return 0
            return 1
        
        if random.random() < self.epsilon:
            return randomAct()

        qValues = self.net(state)
        if qValues[0] < qValues[1]:
            return 1
        elif qValues[0] > qValues[1]:
            return 0
        else:
            return randomAct()

    def train(self, numIters = 20000, epsilon = 0.1, discount = 1, batch_size = 32,
              lr = 0.1, num_experience = 50000, epsilonDecay = False, lrDecay = False,
              evalPerIters = 250, numItersEval = 1000, seed = 0, resume = False):
        '''
        Trains the agent.
        
        Args:
            numIters (int): The number of training iterations.
            epsilon (float): The epsilon value.
            discount (float): The discount factor.
            batch_size (int): The size of a minibatch.
            lr (float): The learning rate.
            num_experience (int): The size of the Experience Replay memory.
            epsilonDecay (bool): Whether to use epsilon decay.
            lrDecay (bool): Whether to use learning rate decay.
            evalPerIters (int): The number of iterations between two evaluation calls.
            numItersEval (int): The number of evaluation iterations.
            seed (int): Random seed for PyTorch.
            resume (bool): Whether to resume from checkpoints.
        '''
        self.epsilon = epsilon
        self.initialEpsilon = epsilon
        self.discount = discount
        self.epsilonDecay = epsilonDecay
        self.lrDecay = lrDecay
        self.evalPerIters = evalPerIters
        self.numItersEval = numItersEval

        self.hparams = HParams(batch_size = batch_size , lr = lr, seed = seed, 
                               num_experience = num_experience, resume = resume)
        if self.hparams.seed != 0:
            torch.manual_seed(self.hparams.seed)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr = self.hparams.lr)
        self.experienceReplay = ExperienceReplay(self.hparams.num_experience)

        self.env.seed(random.randint(0, 100))
        self.net.train()

        reward = 0
        done = False
        maxScore = 0
        maxReward = 0
        counter = 0
        
        for i in range(numIters):
            if i % 50 == 0 or i == numIters - 1:
                print("Iter: ", i)
            
            self.epsilon = self.initialEpsilon / (i + 1) if self.epsilonDecay \
                           else self.initialEpsilon
            self.net.train()
            score = 0
            totalReward = 0
            ob = self.env.reset()
            
            # Performs a dummy action.
            state, _, _, _ = self.env.step(1)
            
            while True:
                counter += 1
                action = self.act(state)
                nextState, reward, done, _ = self.env.step(action)
#                self.env.render()  # Uncomment it to display graphics.
                totalReward += reward
                if reward >= 1:
                    score += 1
                reward = torch.Tensor([reward]).to(self.device)
                survived = 1 - torch.Tensor([done]).to(self.device)
                action = self.actionEncoding[action]
                self.experienceReplay.memorize((state, action, reward, nextState, survived))
                state = nextState
                
                # Fills the memory with at least 5000 observations before training.
                if counter >= 5000: 
                    batch = self.experienceReplay.getBatch(self.hparams.batch_size)
                    if batch:
                        self.updateWeights(batch)

                if done:
                    break
            
            if score > maxScore: maxScore = score
            if totalReward > maxReward: maxReward = totalReward         

            if (i + 1) % self.evalPerIters == 0:
                output = self.test(numIters = self.numItersEval)
                self.saveOutput(output, i + 1)

        self.env.close()
        print("Max Score: ", maxScore)
        print("Max Reward: ", maxReward)
        print()
        
    def test(self, numIters = 20000):
        '''
        Evaluates the agent.
        
        Args:
            numIters (int): The number of evaluation iterations.
        
        Returns:
            dict: A set of scores.
        '''
        self.epsilon = 0
        self.env.seed(0)
        self.net.eval()

        reward = 0
        done = False
        maxScore = 0
        maxReward = 0
        output = defaultdict(int)
        
        with torch.no_grad():
            for i in range(numIters):       
                score = 0
                totalReward = 0
                ob = self.env.reset()
                
                # Performs a dummy action.
                state, _, _, _ = self.env.step(1)
                
                while True:
                    action = self.act(state)
                    state, reward, done, _ = self.env.step(action)
#                    self.env.render()  # Uncomment it to display graphics.
                    totalReward += reward
                    if reward >= 1:
                        score += 1
                    if done:
                        break
                
                output[score] += 1
                if score > maxScore: maxScore = score
                if totalReward > maxReward: maxReward = totalReward
    
        self.env.close()
        print("Max Score: ", maxScore)
        print("Max Reward: ", maxReward)
        print()
        return output
        
    def updateWeights(self, batch):
        '''
        Updates the weights of the network based on a minibatch of observations.
        
        Args:
            batch (zip): A minibatch of observations.
        '''
        stateBatch, actionBatch, rewardBatch, nextStateBatch, survivedBatch = batch
        stateBatch = torch.cat(stateBatch)
        actionBatch = torch.cat(actionBatch)
        rewardBatch = torch.cat(rewardBatch)
        nextStateBatch = torch.cat(nextStateBatch)
        survivedBatch = torch.cat(survivedBatch)

        currQValuesBatch = self.net(stateBatch)
        currQValuesBatch = torch.sum(currQValuesBatch * actionBatch, dim = 1)
        nextQValuesBatch = self.net(nextStateBatch)
        targetQValuesBatch = rewardBatch + self.discount * survivedBatch * \
                             torch.max(nextQValuesBatch, dim = 1).values

        loss = self.criterion(currQValuesBatch, targetQValuesBatch)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step() 

    def saveOutput(self, output, iter):
        '''
        Saves the scores.
        
        Args:
            output (dict): A set of scores.
            iter (int): Current iteration.
        '''
        if not os.path.isdir('scores'):
            os.mkdir('scores')
        with open('./scores/scores_{}.json'.format(iter), 'w') as fp:
            json.dump(output, fp)
            
    def saveModel(self):
        ''' Saves the network. '''
        torch.save(self.net.state_dict(), "model.params")
        
    def loadModel(self):
        ''' Loads the network. '''
        self.net = Net()
        self.net.load_state_dict(torch.load("model.params"))
        self.net = self.net.to(self.device)

