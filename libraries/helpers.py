#==============================================================================
# -*- coding: utf-8 -*-
# Created on Tue Apr 10 17:21:58 2018
# @author: s.jayanthi
# Activation Function and their derivatives
#==============================================================================
import numpy as np

def flatten(x, x_shape):
    flatten_dim = 1;
    for dim in x_shape:
        flatten_dim*=dim;
    x = np.reshape(x, (flatten_dim,1));
    return x; 
def unflatten(x, x_shape):
    x = np.reshape(x, x_shape);
    return x; 