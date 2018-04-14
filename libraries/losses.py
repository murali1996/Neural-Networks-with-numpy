#==============================================================================
# -*- coding: utf-8 -*-
# Created on Tue Apr 10 12:14:57 2018
# @author: s.jayanthi
# Loss Functions and their derivatives
#==============================================================================

import numpy as np
from libraries import activations
from libraries import config
from libraries import helpers
data_type = config.data_type;


x=np.zeros((2,3,4));
y=np.zeros((2,3,4));
shape_temp = x.shape;
flatten_dim = 1;
for dim in shape_temp:
    flatten_dim*=dim;
x = np.reshape(x,(flatten_dim,1))

def rmse(y_pred, y_true, derivative=False):
    # Input
    y_pred = getattr(np, data_type)(y_pred)  #np.float32(y_pred)
    y_true = getattr(np, data_type)(y_true)  #np.float32(y_true)
    # Assertion
    try:
        assert(y_pred.shape==y_true.shape)
    except:
        raise Exception('Output shape predicted IS NOT SAME AS Output shape targetted!');
    # Flatten
    y_pred_shape, y_true_shape = y_pred.shape, y_true.shape
    y_pred_flatten = getattr(helpers, 'flatten')(y_pred, y_pred_shape)
    y_true_flatten = getattr(helpers, 'flatten')(y_true, y_true_shape)
    # Compute
    l_x = (y_pred_flatten-y_true_flatten);
    d_x = l_x*2;
    d_x = getattr(helpers, 'unflatten')(d_x, y_pred_shape)
    if not derivative:
        return np.sum(l_x*l_x)/y_true_flatten.shape[0];
    else:
        return d_x;
def mae(y_pred, y_true, derivative=False):
    # Input
    y_pred = getattr(np, data_type)(y_pred)  #np.float32(y_pred)
    y_true = getattr(np, data_type)(y_true)  #np.float32(y_true)
    # Assertion
    try:
        assert(y_pred.shape==y_true.shape)
    except:
        raise Exception('Output shape predicted IS NOT SAME AS Output shape targetted!');
    # Flatten
    y_pred_shape, y_true_shape = y_pred.shape, y_true.shape
    y_pred_flatten = getattr(helpers, 'flatten')(y_pred, y_pred_shape)
    y_true_flatten = getattr(helpers, 'flatten')(y_true, y_true_shape)
    # Compute
    l_x = (y_pred_flatten-y_true_flatten);
    d_x = l_x.copy(); d_x[l_x>=0] = 1; d_x[l_x<0] = -1;
    d_x = getattr(helpers, 'unflatten')(d_x, y_pred_shape)
    if not derivative:
        return np.sum(np.abs(l_x))/y_true_flatten.shape[0];
    else:
        return d_x;
def softmax_with_cross_entropy(x, y_true, derivative=False):
    # TIP: Use after Softmax activation function only for best results
    # Input
    x = getattr(np, data_type)(x)
    y_true = getattr(np, data_type)(y_true)  #np.float32(y_true)
    # Flatten 
    x_shape, y_true_shape = x.shape, y_true.shape;
    x_flatten = getattr(helpers, 'flatten')(x, x_shape)
    y_pred_flatten = getattr(activations, 'softmax')(x_flatten)
    y_true_flatten = getattr(helpers, 'flatten')(y_true, y_true_shape)
    # Assertion
    try:
        assert(y_pred_flatten.shape==y_true_flatten.shape)
    except:
        raise Exception('Output shape predicted IS NOT SAME AS Output shape targetted!');
    # Compute
    l_x = np.array([np.log(a) for a in y_pred_flatten], dtype=data_type);
    d_x = y_pred_flatten-y_true_flatten;
    d_x = getattr(helpers, 'unflatten')(d_x, x_shape)
    if not derivative:
        return np.sum(-1*y_true*l_x)/y_true_flatten.shape[0];
    else:
        return d_x;