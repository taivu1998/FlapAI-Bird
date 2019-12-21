'''
This program implements a Function Approximation Agent with Linear Regression.
'''

import os, sys
sys.path.append('../game')
sys.path.append('../utils')

from collections import defaultdict
import json
import random
import numpy as np

import gym
from TemplateAgent import FlappyBirdAgent
from utils import *
from FlappyBirdGame import FlappyBirdLR


class FuncApproxLRAgent(FlappyBirdAgent):
    ''' Function Approximation Agent with Linear Regression. '''
    
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
        self.weights = defaultdict(float)
        self.env = FlappyBirdLR(gym.make('FlappyBird-v0'))
    
    def featureExtractor(self, state, action):
        '''
        Extracts features from a tuple (state, action).
        
        Args:
            state (dict): A state.
            action (int): 0 or 1.
        
        Returns:
            dict: A feature vector.
        '''
        hor_dist_to_next_pipe = state['next_pipe_dist_to_player']
        ver_dist_to_next_pipe = state['next_pipe_bottom_y'] - state['player_y']
        result = defaultdict(float)
        result['player_vel'] = state['player_vel'] / 10
        result['hor_dist_to_next_pipe'] = hor_dist_to_next_pipe / 288
        result['ver_dist_to_next_pipe'] = ver_dist_to_next_pipe / 512
#        result['hor_dist_to_next_pipe'] = (hor_dist_to_next_pipe / 288)**3
#        result['ver_dist_to_next_pipe'] = 1/ver_dist_to_next_pipe if ver_dist_to_next_pipe != 0 else 1
        result['action'] = action
        result['bias'] = 1
        return result
    
    def qValues(self, state, action):
        '''
        Returns the current Q-value for a tuple (state, action).
        
        Args:
            state (dict): A state.
            action (int): 0 or 1.
        
        Returns:
            float: A Q-value.
        '''
        return dotProduct(self.weights, self.featureExtractor(state, action))
        
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

        qValues = [self.qValues(state, action) for action in self.actions]
        if qValues[0] < qValues[1]:
            return 1
        elif qValues[0] > qValues[1]:
            return 0
        else:
            return randomAct()
    
    def saveWeights(self):
        ''' Saves the weights. '''
        with open('weights.json', 'w') as fp:
            json.dump(self.weights, fp)

    def loadWeights(self):
        ''' Loads the weights. '''
        with open('weights.json') as fp:
            self.weights = json.load(fp)
    
    def train(self, order = 'forward', numIters = 20000, epsilon = 0.1, discount = 1,
              eta = 0.01, epsilonDecay = False, etaDecay = False, evalPerIters = 250,
              numItersEval = 1000):
        '''
        Trains the agent.
        
        Args:
            order (str): The order of updates, 'forward' or 'backward'.
            numIters (int): The number of training iterations.
            epsilon (float): The epsilon value.
            discount (float): The discount factor.
            eta (float): The eta value.
            epsilonDecay (bool): Whether to use epsilon decay.
            etaDecay (bool): Whether to use eta decay.
            evalPerIters (int): The number of iterations between two evaluation calls.
            numItersEval (int): The number of evaluation iterations.
        '''
        self.epsilon = epsilon
        self.initialEpsilon = epsilon
        self.discount = discount
        self.epsilonDecay = epsilonDecay
        self.etaDecay = etaDecay
        self.eta = eta
        self.etaDecay = etaDecay
        self.evalPerIters = evalPerIters
        self.numItersEval = numItersEval
        self.env.seed(random.randint(0, 100))

        reward = 0
        done = False
        maxScore = 0
        maxReward = 0
        
        for i in range(numIters):
            if i % 50 == 0 or i == numIters - 1:
                print("Iter: ", i)
            
            self.epsilon = self.initialEpsilon / (i + 1) if self.epsilonDecay \
                           else self.initialEpsilon
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
                    self.updateWeight(state, action, reward, nextState)
            else:
                for (state, action, reward, nextState) in gameIter[::-1]:
                    self.updateWeight(state, action, reward, nextState)
            
            if self.etaDecay:
                self.eta *= (i + 1) / (i + 2)
            
            if (i + 1) % self.evalPerIters == 0:
                output = self.test(numIters = self.numItersEval)
                self.saveOutput(output, i + 1)
                self.saveWeights()
                
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

        reward = 0
        done = False
        maxScore = 0
        maxReward = 0
        output = defaultdict(int)
        
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
            
    def updateWeight(self, state, action, reward, nextState):
        '''
        Updates the weights based on an observation.
        
        Args:
            state, nextState (str): Two states.
            action (int): 0 or 1.
            reward (int): The reward value.
        '''
        nextQValues = [self.qValues(nextState, nextAction) for nextAction in self.actions]
        nextV = max(nextQValues)
        currQValue = self.qValues(state, action)
        currFeatures = self.featureExtractor(state, action)
        increment(self.weights, -self.eta * (currQValue - reward - self.discount * nextV), currFeatures)

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

