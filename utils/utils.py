'''
This file contains several helper funcions.
'''

import math


def discretize(num, rounding):
    ''' Discretize the input num base on the value rounding. '''
    return rounding * math.floor(num / rounding)


def dotProduct(v1, v2):
    '''
    Computes the dot product between two feature vectors v1 and v2.
    This is a function from Assignment 1 of CS 221.
    '''
    if len(v1) < len(v2):
        return dotProduct(v2, v1)
    return sum(v1.get(key, 0) * value for key, value in v2.items())


def increment(v1, scale, v2):
    '''
    Executes v1 += scale * v2 for feature vectors.
    This is a function from Assignment 1 of CS 221.
    '''
    for key, value in v2.items():
        v1[key] = v1.get(key, 0) + value * scale
