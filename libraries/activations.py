#==============================================================================
# -*- coding: utf-8 -*-
# Created on Tue Apr 10 11:15:29 2018
# @author: s.jayanthi
# Activation Function and their derivatives
#==============================================================================
import numpy as np
from libraries import config
data_type = config.data_type;

def relu(x, derivative=False):
    x = getattr(np, data_type)(x)  #np.float32(x)
    if not derivative:
        f_x = x.copy(); f_x[x<0] = 0;
        return f_x;
    else:
        d_x = x.copy(); d_x[x>0]=1;
        return d_x;
def sigmoid(x, derivative=False):
    x = getattr(np, data_type)(x)  #np.float32(x)
    f_x = np.array([1/(1+np.exp(-a)) for a in x], dtype=data_type)
    if not derivative:
        return f_x;
    else:
        return f_x*(1-f_x);
def tanh(x, derivative=False):
    x = getattr(np, data_type)(x)  #np.float32(x)
    f_x = np.array([(np.exp(a)-np.exp(-a))/(np.exp(a)+np.exp(-a)) for a in x], dtype=data_type)
    if not derivative:
        return f_x;
    else:
        return 1-f_x*f_x;
def softmax(x, derivative=False):
    x = getattr(np, data_type)(x)  #np.float32(x)
    max_ = np.max(x);
    den_ = np.sum([np.exp(a-max_) for a in x]) #To avoid overfitting
    f_x = np.array([np.exp(a-max_)/den_ for a in x], dtype=data_type)
    if not derivative:
        return f_x;
    else:
        return f_x*(1-f_x);
