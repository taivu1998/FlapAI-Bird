'''
This program implements a Baseline Agent, which follows a random policy.
'''

import os, sys
sys.path.append('../game')
sys.path.append('../utils')

from collections import defaultdict
import json
import random

import gym
from TemplateAgent import FlappyBirdAgent
from FlappyBirdGame import FlappyBirdNormal

import warnings
warnings.filterwarnings('ignore')


class BaselineAgent(FlappyBirdAgent):
    ''' Baseline Agent with a random policy. '''
    
    def __init__(self, actions, probFlap = 0.5):
        '''
        Initializes the agent.
        
        Args:
            actions (list): Possible action values.
            probFlap (float): The probability of flapping when choosing
                              the next action randomly.
        '''
        super().__init__(actions)
        self.probFlap = probFlap
        self.env = FlappyBirdNormal(gym.make('FlappyBird-v0'))

    def act(self, state):
        '''
        Returns the next action for the current state.
        
        Args:
            state (list): The current state.
            
        Returns:
            int: 0 or 1.
        '''
        if random.random() < self.probFlap:
            return 0
        return 1
    
    def train(self, numIters = 20000, evalPerIters = 250, numItersEval = 1000):
        '''
        Trains the agent.
        
        Args:
            numIters (int): The number of training iterations.
            evalPerIters (int): The number of iterations between two evaluation calls.
            numItersEval (int): The number of evaluation iterations.
        '''
        print("No training needed!")
        
        self.evalPerIters = evalPerIters
        self.numItersEval = numItersEval
        for i in range(numIters):
            if i % 50 == 0 or i == numIters - 1:
                print("Iter: ", i)
            
            if (i + 1) % self.evalPerIters == 0:
                output = self.test(numIters = self.numItersEval)
                self.saveOutput(output, i + 1)
       
    def test(self, numIters = 2000):
        '''
        Evaluates the agent.
        
        Args:
            numIters (int): The number of evaluation iterations.
        
        Returns:
            dict: A set of scores.
        '''
        self.env.seed(0)

        done = False
        maxScore = 0
        maxReward = 0
        output = defaultdict(int)
        counter = 0
        
        for i in range(numIters):
            score = 0
            totalReward = 0
            ob = self.env.reset()
            state = self.env.getGameState()
            
            while True:
                action = self.act(state)
                state, reward, done, _ = self.env.step(action)
#                self.env.render()  # Uncomment it to display graphics.
                totalReward += reward
                if reward >= 1:
                    score += 1
                counter += 1
                if done:
                    break
                    break
                    
            output[score] += 1
            if score > maxScore: maxScore = score
            if totalReward > maxReward: maxReward = totalReward
    
        self.env.close()
        print("Max Score Test: ", maxScore)
        print("Max Reward Test: ", maxReward)
        print()
        return output

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
