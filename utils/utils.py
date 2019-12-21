'''
This file contains several helper funcions.
'''

import math


def discretize(num, rounding):
    '''
    Discretizes the input num base on the value rounding.
    
    Args:
        num (int): An input value.
        rounding (int): The level of discretization.
        
    Returns:
        int: A discretized output value.
    '''
    return rounding * math.floor(num / rounding)


def dotProduct(v1, v2):
    '''
    Computes the dot product between two feature vectors v1 and v2.
    
    Args:
        v1, v2 (dict): Two input vectors.
        
    Returns:
        dict: A dot product.
    '''
    if len(v1) < len(v2):
        return dotProduct(v2, v1)
    return sum(v1.get(key, 0) * value for key, value in v2.items())


def increment(v1, scale, v2):
    '''
    Executes v1 += scale * v2 for feature vectors.
    
    Args:
        v1, v2 (dict): Two input vectors.
        scale (float): A scale value.
    '''
    for key, value in v2.items():
        v1[key] = v1.get(key, 0) + value * scale
