#==============================================================================
# -*- coding: utf-8 -*-
# Created on Tue Apr 10 14:57:54 2018
# @author: s.jayanthi
# Activation Function and their derivatives
#==============================================================================
import numpy as np
from libraries import config

def input_layer(input_shape, positional_args):
    output_shape = positional_args[1];
    output_shape = np.expand_dims(output_shape, axis=-1)
    input_shape, weights, biases, activation_type = None, None, None, None
    return input_shape, output_shape, weights, biases, activation_type
def dense_layer(input_shape, positional_args):
    output_shape = positional_args[1];
    output_shape = np.expand_dims(output_shape, axis=-1)
    n_weights = output_shape[0]*input_shape[0]
    weights = np.random.randn(n_weights)/np.sqrt(n_weights);
    weights = np.reshape(weights, (input_shape[0],output_shape[0])).astype(config.data_type);
    biases = np.zeros(output_shape[0], dtype=config.data_type);
    if len(positional_args)<3:
        activation_type = 'relu';
    else:
        activation_type = positional_args[2];
    return input_shape, output_shape, weights, biases, activation_type
def output_layer(input_shape, positional_args):
    output_shape = positional_args[1];
    output_shape = np.expand_dims(output_shape, axis=-1)
    n_weights = output_shape[0]*input_shape[0]
    weights = np.random.randn(n_weights)/np.sqrt(n_weights);
    weights = np.reshape(weights, (input_shape[0],output_shape[0])).astype(config.data_type);
    biases = np.zeros(output_shape[0], dtype=config.data_type);
    if len(positional_args)<3:
        activation_type = None;
    else:
        activation_type = positional_args[2];
    return input_shape, output_shape, weights, biases, activation_type    
def loss_layer(input_shape, positional_args):
    
    return