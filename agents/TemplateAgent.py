'''
This program implements an abstract class, which acts as a template for
all Flappy Bird Agents.
'''

class FlappyBirdAgent(object):
    ''' Template Agent. '''
    
    def __init__(self, actions):
        '''
        Initializes the agent.
        
        Args:
            actions (list): Possible action values.
        '''
        self.actions = actions

    def act(self, state):
        '''
        Returns the next action for the current state.
        
        Args:
            state (list): The current state.
        '''
        raise NotImplementedError("Override this.")
    
    def train(self, numIters):
        '''
        Trains the agent.
        
        Args:
            numIters (int): The number of training iterations.
        '''
        raise NotImplementedError("Override this.")
    
    def test(self, numIters):
        '''
        Evaluates the agent.
        
        Args:
            numIters (int): The number of evaluation iterations.
        '''
        raise NotImplementedError("Override this.")
        
    def saveOutput(self):
        ''' Saves the scores. '''
        raise NotImplementedError("Override this.")
