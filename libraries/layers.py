#==============================================================================
# -*- coding: utf-8 -*-
# Created on Tue Apr 10 14:57:54 2018
# @author: s.jayanthi
# Activation Function and their derivatives
#==============================================================================
import numpy as np
from libraries import config

def input_layer(thisLayer, keyword_args):
    try:
        thisLayer['output_shape'] = keyword_args['shape'];
    except KeyError:
        raise Exception('shape parameter undefined')
    try:
        assert(thisLayer['output_shape']!=None);
    except AssertionError:
        raise Exception('shape specified for input_layer cannot be None')
    weights, biases = None, None
    return thisLayer, weights, biases
def dense_layer(thisLayer, keyword_args):
    try:
        thisLayer['output_shape'] = keyword_args['shape'];
    except KeyError:
        raise Exception('shape parameter undefined')
    try:
        assert(thisLayer['output_shape']!=None);
    except AssertionError:
        raise Exception('shape specified for input_layer cannot be None')
    try:
        assert(len(thisLayer['output_shape'])==len(thisLayer['input_shape'])==1);
    except AssertionError:
        raise Exception('shape for dense layer can be (<dim>,) type');
    try:
        thisLayer['activation_type'] = keyword_args['activation_type'];
    except KeyError:
        pass;
    try:
        thisLayer['dropout'] = keyword_args['dropout'];
        if thisLayer['dropout']>=1 or thisLayer['dropout']<0:
            raise Exception('**dropout** can only be in the range [0,1)')
    except KeyError:
        pass;
    n_weights = thisLayer['input_shape'][0]*thisLayer['output_shape'][0];
    weights = np.random.randn(n_weights)/np.sqrt(n_weights);
    weights = np.reshape(weights, (thisLayer['input_shape'][0],thisLayer['output_shape'][0])).astype(config.data_type);
    biases = np.zeros((thisLayer['output_shape'][0],1), dtype=config.data_type);
    thisLayer['any_params'] = True
    thisLayer['trainable'] = True;
    return thisLayer, weights, biases
def conv2D_layer(thisLayer, keyword_args):
    try:
        thisLayer['kernel_shape'] = keyword_args['kernel_shape'];
        try:
            assert(len(thisLayer['stride'])==2)
        except AssertionError:
            raise Exception('**kernel_shape**  possible shape for conv2D = (<dim1>,<dim2>)')
    except KeyError: #'kernel_shape has to be defined; default is <(3,3)>'
        pass;
    try:
        thisLayer['n_kernels'] = keyword_args['n_kernels'];
    except KeyError:
        raise Exception('n_kernels has to be defined')
    try:
        thisLayer['padding'] = keyword_args['padding'];
        if  thisLayer['padding'] is not 'same' and thisLayer['padding'] is not 'valid':
            raise Exception('**padding** can be one of **same** or **valid**');
    except KeyError: #'padding has to be defined; default is <same>'
        pass;
    try:
        thisLayer['stride'] = keyword_args['stride'];
        try:
            assert(len(thisLayer['stride'])==2)
            if thisLayer['stride'][0]>thisLayer['kernel_shape'][0] or thisLayer['stride'][1]>thisLayer['kernel_shape'][1]:
                raise Exception('Along all axis, value of stride must be less than or equal to kernel_shape');
        except AssertionError:
            raise Exception('With padding **valid** or **same**, only compatible stride shape = (<dim1>,<dim2>)')  
    except KeyError:
        pass;  
    try:
        thisLayer['activation_type'] = keyword_args['activation_type'];
    except KeyError:
        pass;
    try:
        thisLayer['dropout'] = keyword_args['dropout'];
        if thisLayer['dropout']>=1 or thisLayer['dropout']<0:
            raise Exception('**dropout** can only be in the range [0,1)')
    except KeyError:
        pass;
    if thisLayer['padding']=='same':
        try:
            assert(thisLayer['stride']==(1,1))
        except AssertionError:
            raise Exception('With padding **same**, only compatible stride=(1,1)')
        thisLayer['output_shape'] =  list(thisLayer['input_shape']);
        thisLayer['output_shape'][-1] = thisLayer['n_kernels'];
        thisLayer['output_shape'] = tuple(thisLayer['output_shape']);
    elif thisLayer['padding']=='valid':
        thisLayer['output_shape'] =  list(thisLayer['input_shape']);
        thisLayer['output_shape'][0] = int(np.round((thisLayer['input_shape'][0]-1-thisLayer['kernel_shape'][0])/thisLayer['stride'][0]))+1;
        thisLayer['output_shape'][1] = int(np.round((thisLayer['input_shape'][1]-1-thisLayer['kernel_shape'][1])/thisLayer['stride'][1]))+1;
        thisLayer['output_shape'][2] = thisLayer['n_kernels'];  
        thisLayer['output_shape'] = tuple(thisLayer['output_shape']);
    n_weights = thisLayer['n_kernels']*thisLayer['kernel_shape'][0]*thisLayer['kernel_shape'][1]*thisLayer['input_shape'][2];
    weights = np.random.randn(n_weights)/np.sqrt(n_weights);
    weights = np.reshape(weights, (thisLayer['n_kernels'],thisLayer['kernel_shape'][0],thisLayer['kernel_shape'][1],thisLayer['input_shape'][2])).astype(config.data_type);
    biases = np.zeros((thisLayer['n_kernels'],1), dtype=config.data_type); 
    thisLayer['any_params'] = True
    thisLayer['trainable'] = True;
    return thisLayer, weights, biases
def flatten_layer(thisLayer, keyword_args):
    temp = 1;
    for dim in thisLayer['input_shape']:
        temp*=dim;
    thisLayer['output_shape'] = (temp,);
    weights, biases = None, None
    return thisLayer, weights, biases  