#==============================================================================
# -*- coding: utf-8 -*-
# Created on Tue Apr 10 12:14:57 2018
# @author: s.jayanthi
# Loss Functions and their derivatives
#==============================================================================

import numpy as np
from libraries import activations
from libraries import config
data_type = config.data_type;

def rmse(y_pred, y_true, derivative=False):
    y_pred = getattr(np, data_type)(y_pred)  #np.float32(y_pred)
    y_true = getattr(np, data_type)(y_true)  #np.float32(y_true)
    l_x = (y_pred-y_true);
    if not derivative:
        return np.sum(l_x*l_x)/y_pred.shape[0];
    else:
        return l_x*2;
def mae(y_pred, y_true, derivative=False):
    y_pred = getattr(np, data_type)(y_pred)  #np.float32(y_pred)
    y_true = getattr(np, data_type)(y_true)  #np.float32(y_true)
    l_x = (y_pred-y_true);
    if not derivative:
        return np.sum(np.abs(l_x))/y_pred.shape[0];
    else:
        d_x = l_x.copy();
        d_x[l_x>=0] = 1; d_x[l_x<0] = -1;
        return d_x;
def softmax_with_cross_entropy(x, y_true, derivative=False):
    # TIP: Use after Softmax activation function only for best results
    y_pred = getattr(activations, 'softmax')(x)
    y_true = getattr(np, data_type)(y_true)  #np.float32(y_true)
    l_x = np.array([np.log(a) for a in y_pred], dtype=data_type)
    if not derivative:
        return np.sum(-1*y_true*l_x)/y_pred.shape[0];
    else:
        return y_pred-y_true;