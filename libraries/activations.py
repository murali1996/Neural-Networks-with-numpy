#==============================================================================
# -*- coding: utf-8 -*-
# Created on Tue Apr 10 11:15:29 2018
# @author: s.jayanthi
# Activation Function and their derivatives
#==============================================================================
import numpy as np
from libraries import config
from libraries import helpers

def relu(x, derivative=False):
    # Input
    x = getattr(np, config.data_type)(x)  #np.float32(x)
    # Flatten
    x_shape = x.shape;
    x_flatten = getattr(helpers, 'flatten')(x, x_shape);
    # Compute
    if not derivative:
        f_x = x_flatten.copy(); f_x[x_flatten<0] = 0;
        f_x = getattr(helpers, 'unflatten')(f_x, x_shape);
        return f_x;
    else:
        d_x = x_flatten.copy(); d_x[x_flatten>0]=1; d_x[x_flatten<=0]=0;
        d_x = getattr(helpers, 'unflatten')(d_x, x_shape);
        return d_x;
def sigmoid(x, derivative=False):
    # Input
    x = getattr(np, config.data_type)(x)  #np.float32(x)
    # Flatten
    x_shape = x.shape;
    x_flatten = getattr(helpers, 'flatten')(x, x_shape);
    # Compute
    if not derivative:
        f_x = np.array([1/(1+np.exp(-a)) for a in x_flatten], dtype=config.data_type);
        f_x = getattr(helpers, 'unflatten')(f_x, x_shape);
        return f_x;
    else:
        f_x = np.array([1/(1+np.exp(-a)) for a in x_flatten], dtype=config.data_type);
        d_x = f_x*(1-f_x);
        d_x = getattr(helpers, 'unflatten')(d_x, x_shape);
        return d_x;
def tanh(x, derivative=False):
    # Input
    x = getattr(np, config.data_type)(x)  #np.float32(x)
    # Flatten
    x_shape = x.shape;
    x_flatten = getattr(helpers, 'flatten')(x, x_shape);
    # Compute
    if not derivative:
        f_x = np.array([(np.exp(a)-np.exp(-a))/(np.exp(a)+np.exp(-a)) for a in x_flatten], dtype=config.data_type)
        f_x = getattr(helpers, 'unflatten')(f_x, x_shape);
        return f_x;
    else:
        f_x = np.array([(np.exp(a)-np.exp(-a))/(np.exp(a)+np.exp(-a)) for a in x_flatten], dtype=config.data_type)
        d_x = 1-f_x*f_x;
        d_x = getattr(helpers, 'unflatten')(d_x, x_shape);
        return d_x;
def softmax(x, derivative=False):
    # Input
    x = getattr(np, config.data_type)(x)  #np.float32(x)
    # Flatten
    x_shape = x.shape;
    x_flatten = getattr(helpers, 'flatten')(x, x_shape);
    # Compute
    max_ = np.max(x_flatten);
    den_ = np.sum([np.exp(a-max_) for a in x_flatten]) #To avoid overfitting
    if not derivative:
        f_x = np.array([np.exp(a-max_)/den_ for a in x_flatten], dtype=config.data_type)
        f_x = getattr(helpers, 'unflatten')(f_x, x_shape);
        return f_x;
    else:
        f_x = np.array([np.exp(a-max_)/den_ for a in x_flatten], dtype=config.data_type)
        d_x  = f_x*(1-f_x)
        d_x = getattr(helpers, 'unflatten')(d_x, x_shape);
        return d_x;
