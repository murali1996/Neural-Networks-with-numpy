#==============================================================================
# -*- coding: utf-8 -*-
# Created on Tue Apr 10 17:21:58 2018
# @author: s.jayanthi
# Activation Function and their derivatives
#==============================================================================
import numpy as np
from libraries import config

def flatten(x, x_shape):
    flatten_dim = 1;
    for dim in x_shape:
        flatten_dim*=dim;
    x = np.reshape(x, (flatten_dim,1));
    return x; 
def unflatten(x, x_shape):
    x = np.reshape(x, x_shape);
    return x; 
def padder(x, new_height, new_width):
    height, width, channels = x.shape;
    new_rows = np.zeros((new_height,width,channels),dtype=config.data_type);
    x = np.concatenate((x,new_rows), axis=-3);
    height, width, channels = x.shape;
    new_cols = np.zeros((height,new_width,channels),dtype=config.data_type);
    x = np.concatenate((x,new_cols), axis=-2);
    return x;    