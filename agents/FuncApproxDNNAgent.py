'''
This program implements a Function Approximation Agent with a Feed Forward Neural Network.
'''

from __future__ import print_function

import os, sys
sys.path.append('../game')
sys.path.append('../utils')

from collections import defaultdict
import json
import random
import numpy as np

import gym
from TemplateAgent import FlappyBirdAgent
from FlappyBirdGame import FlappyBirdDNN

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


class Net(nn.Module):
    ''' Feed Forward Neural Network. '''
    
    def __init__(self):
        ''' Initializes the network architecture. '''
        super(Net, self).__init__()
        self.linear1 = nn.Linear(3, 50) 
        self.linear2 = nn.Linear(50, 20)
        self.linear3 = nn.Linear(20, 2)

    def forward(self, x):
        ''' Performs a forward pass through the network. '''
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        return x


class HParams():
    ''' A class storing hyperparameters of the model. '''
    
    def __init__(self, lr = 0.1, resume = False, seed = 0, 
                 batch_size = 32, start_epoch = 0, epoch = 10000, 
                 decay = 1e-4, num_experience = 50000):
        ''' Initializes the parameters. '''
        self.lr = lr
        self.resume = resume
        self.seed = seed
        self.start_epoch = start_epoch
        self.epoch = epoch
        self.decay = decay


class FuncApproxDNNAgent(FlappyBirdAgent):
    ''' Function Approximation Agent with a Feed Forward Neural Network. '''
    
    def __init__(self, actions, probFlap = 0.1):
        ''' Initializes the agent. '''
        super().__init__(actions)
        self.probFlap = probFlap
        self.env = FlappyBirdDNN(gym.make('FlappyBird-v0'))
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.net = Net().to(self.device)
        self.criterion = torch.nn.MSELoss()
    

    def qValues(self, state):
        ''' Returns the current Q-values for a state. '''
        input = torch.Tensor(state).to(self.device)
        return self.net(input)
        
        
    def act(self, state):
        ''' Returns the next action for the current state. '''
        def randomAct():
            if random.random() < self.probFlap:
                return 0
            return 1
        
        if random.random() < self.epsilon:
            return randomAct()

        input = torch.Tensor(state).to(self.device)
        qValues = self.net(input)
        if qValues[0] < qValues[1]:
            return 1
        elif qValues[0] > qValues[1]:
            return 0
        else:
            return randomAct()
    
    
    def train(self, order = 'forward', numIters = 20000, epsilon = 0.1, discount = 1, lr = 0.1, epsilonDecay = False,
              lrDecay = False, evalPerIters = 250, numItersEval = 1000, seed = 0, resume = False):
        ''' Trains the agent. '''
        self.epsilon = epsilon
        self.initialEpsilon = epsilon
        self.discount = discount
        self.epsilonDecay = epsilonDecay
        self.lrDecay = lrDecay
        self.evalPerIters = evalPerIters
        self.numItersEval = numItersEval

        self.hparams = HParams(lr = lr, seed = seed, resume = resume)
        if self.hparams.seed != 0:
            torch.manual_seed(self.hparams.seed)
        self.optimizer = torch.optim.SGD(self.net.parameters(), lr = self.hparams.lr)

        self.env.seed(random.randint(0, 100))
        self.net.train()

        reward = 0
        done = False
        maxScore = 0
        maxReward = 0
        
        for i in range(numIters):
            if i % 50 == 0 or i == numIters - 1:
                print("Iter: ", i)
            
            self.epsilon = self.initialEpsilon / (i + 1) if self.epsilonDecay \
                           else self.initialEpsilon
            self.net.train()
            score = 0
            totalReward = 0
            ob = self.env.reset()
            gameIter = []
            state = self.env.getGameState()
            
            while True:
                action = self.act(state)
                nextState, reward, done, _ = self.env.step(action)
                gameIter.append((state, action, reward, nextState))
                state = nextState
#                self.env.render()  # Uncomment it to display graphics.
                totalReward += reward
                if reward >= 1:
                    score += 1
                if done:
                    break
            
            if score > maxScore: maxScore = score
            if totalReward > maxReward: maxReward = totalReward
            
            if order == 'forward':
                for (state, action, reward, nextState) in gameIter:
                    self.updateWeights(state, action, reward, nextState)
            else:
                for (state, action, reward, nextState) in gameIter[::-1]:
                    self.updateWeights(state, action, reward, nextState)

            if (i + 1) % self.evalPerIters == 0:
                output = self.test(numIters = self.numItersEval)
                self.saveOutput(output, i + 1)
                self.saveModel()

        self.env.close()
        print("Max Score: ", maxScore)
        print("Max Reward: ", maxReward)
        print()
        
    
    def test(self, numIters = 20000):
        ''' Evaluates the agent. '''
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
                state = self.env.getGameState()
                
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
        
            
    def updateWeights(self, state, action, reward, nextState):
        ''' Updates the weights of the network based on an observation. '''
        nextQValues = self.qValues(nextState)
        nextV, _ = torch.max(nextQValues, 0)
        currQValue = self.qValues(state)[action]
        targetQValue = reward + self.discount * nextV
        loss = self.criterion(currQValue, targetQValue)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        

    def saveOutput(self, output, iter):
        ''' Save the scores. '''
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
        
