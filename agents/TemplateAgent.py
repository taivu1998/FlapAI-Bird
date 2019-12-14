'''
This program implements an abstract class, which acts as a template for all Flappy Bird Agents.
'''

class FlappyBirdAgent(object):
    ''' Template Agent. '''
    
    def __init__(self, actions):
        ''' Initializes the agent. '''
        self.actions = actions

    def act(self, state):
        ''' Returns the next action for the current state. '''
        raise NotImplementedError("Override this")
    
    def train(self, numIters):
        ''' Trains the agent. '''
        raise NotImplementedError("Override this")
    
    def test(self, numIters):
        ''' Evaluates the agent. '''
        raise NotImplementedError("Override this")
        
    def saveOutput(self):
        ''' Save the scores. '''
        raise NotImplementedError("Override this")
