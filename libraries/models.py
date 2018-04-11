#==============================================================================
# -*- coding: utf-8 -*-
# Created on Tue Apr 10 14:57:54 2018
# @author: s.jayanthi
# Models/Main function
#==============================================================================
import numpy as np
from libraries import layers

class Sequential(object):
    def __init__(self):
        # RunTime
        self.inputs = [];
        self.targets = [];
        self.layer_inputs = [];
        # Initializations
        self.layer_types = [];
        self.layer_output_shapes = [];
        self.layer_weights = [];
        self.layer_biases = [];
        self.activation_type = []; 
        # Loss and optimizers
        self.loss_type = [];
        self.optimizer_type = [];
        return
    def add(self, *positional_args):
        # Know the layer type
        layer_type = positional_args[0];
        self.layer_types.append(layer_type);
        # Know the output dimensions
        out_dims = positional_args[1];
        if self.layer_output_shapes==[]: #First layer 
            input_shape, output_shape, weights, biases, activation_type = \
                getattr(layers, layer_type)(None, out_dims);
        else:
            input_shape, output_shape, weights, biases, activation_type = \
                getattr(layers, layer_type)(self.layer_output_shapes[-1], out_dims);        
        # Know the activation type
        if len(positional_args)>2:
            activation_type = positional_args[2];
            
        # Know more information
            ##Under-Construction

        self.layer_weights.append(weights);
        self.layer_biases.append(biases);
        self.layer_output_shapes.append(output_shape);
        self.activation_type.append(activation_type);
        return
    def compile(self, **keyword_args):
        self.loss_type = keyword_args['loss'];
        self.optimizer_type = keyword_args['optimizer']
        return
#    def train(self, inputs, targets, n_epochs=100):
#        self.inputs = inputs; self.targets = targets;
#        batch_size = inputs.shape[0];
#        batch_start = 0;
#        for epoch in range(n_epochs):
#            while(batch_start+batch_size<=inputs.shape[0]):
#                for ind in range(batch_start, batch_start+batch_size):
#                    # Input-Output Initialization
#                    currInput = inputs[ind];
#                    currTarget = targets[ind];
#                    # Forward Pass
#                    for layer_ind, layer in enumerate(self.layer_types):
#                        if self.layer_weights[layer_ind]!=None:
#                            currInput = np.dot(currInput,self.layer_weights[layer_ind]);
#                            currInput = currInput + self.layer_biases[layer_ind];
#                            if self.activation_type[layer_ind]!=None:
#                                act_type = self.activation_type[layer_ind];
#                        else:
#                            self.layer_inputs.append(currInput);
#                            continue;
#                batch_start+=batch_size;
#        return
    
    
    
    
    
    
    
    
    
    