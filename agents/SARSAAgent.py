'''
This program implements a SARSA Agent.
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
from FlappyBirdGame import FlappyBirdNormal

import warnings
warnings.filterwarnings('ignore')


class SARSAAgent(FlappyBirdAgent):
    ''' SARSA Agent. '''
    
    def __init__(self, actions, probFlap = 0.5, rounding = None):
        '''
        Initializes the agent.
        
        Args:
            actions (list): Possible action values.
            probFlap (float): The probability of flapping when choosing
                              the next action randomly.
            rounding (int): The level of discretization.
        '''
        super().__init__(actions)
        self.probFlap = probFlap
        self.qValues = defaultdict(float)
        self.env = FlappyBirdNormal(gym.make('FlappyBird-v0'), rounding = rounding)

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
        
        qValues = [self.qValues.get((state, action), 0) for action in self.actions]
            
        if qValues[0] < qValues[1]:
            return 1
        elif qValues[0] > qValues[1]:
            return 0
        else:
            return randomAct()
            
    def saveQValues(self):
        ''' Saves the Q-values. '''
        toSave = {key[0] + ' action ' + str(key[1]) : self.qValues[key] for key in self.qValues}
        with open('qValues.json', 'w') as fp:
            json.dump(toSave, fp)
            
    def loadQValues(self):
        ''' Loads the Q-values. '''
        def parseKey(key):
            state = key[:-9]
            action = int(key[-1])
            return (state, action)

        with open('qValues.json') as fp:
            toLoad = json.load(fp)
            self.qValues = {parseKey(key) : toLoad[key] for key in toLoad}

    def train(self, order = 'forward', numIters = 20000, epsilon = 0.1, discount = 1,
              eta = 0.9, epsilonDecay = False, etaDecay = False, evalPerIters = 250,
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
        self.eta = eta
        self.epsilonDecay = epsilonDecay
        self.etaDecay = etaDecay
        self.evalPerIters = evalPerIters
        self.numItersEval = numItersEval
        self.env.seed(random.randint(0, 100))

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
            action = self.act(state)
            
            while True:
                nextState, reward, done, _ = self.env.step(action)
                nextAction = self.act(nextState)
                gameIter.append((state, action, reward, nextState, nextAction))
                state = nextState
                action = nextAction
#                self.env.render()  # Uncomment it to display graphics.
                totalReward += reward
                if reward >= 1:
                    score += 1
                if done:
                    break
            
            if score > maxScore: maxScore = score
            if totalReward > maxReward: maxReward = totalReward
            
            if order == 'forward':
                for (state, action, reward, nextState, nextAction) in gameIter:
                    self.updateQ(state, action, reward, nextState, nextAction)
            else:
                for (state, action, reward, nextState, nextAction) in gameIter[::-1]:
                    self.updateQ(state, action, reward, nextState, nextAction)
                
            if self.etaDecay:
                self.eta *= (i + 1) / (i + 2)

            if (i + 1) % self.evalPerIters == 0:
                output = self.test(numIters = self.numItersEval)
                self.saveOutput(output, i + 1)
                self.saveQValues()
                
        self.env.close()
        print("Max Score Train: ", maxScore)
        print("Max Reward Train: ", maxReward)
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
#                self.env.render()  # Uncomment it to display graphics.
                totalReward += reward
                if reward >= 1:
                    score += 1
                if done:
                    break
                    
            output[score] += 1
            if score > maxScore: maxScore = score
            if totalReward > maxReward: maxReward = totalReward
    
        self.env.close()
        print("Max Score Test: ", maxScore)
        print("Max Reward Test: ", maxReward)
        print()
        return output
         
    def updateQ(self, state, action, reward, nextState, nextAction):
        '''
        Updates the Q-values based on an observation.
        
        Args:
            state, nextState (str): Two states.
            action, nextAction (int): 0 or 1.
            reward (int): The reward value.
        '''
        self.qValues[(state, action)] = (1 - self.eta) * self.qValues.get((state, action), 0) \
                                        + self.eta * (reward + self.discount * self.qValues.get((nextState, nextAction), 0))
        
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

