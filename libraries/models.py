#==============================================================================
# -*- coding: utf-8 -*-
# Created on Tue Apr 10 14:57:54 2018
# @author: s.jayanthi
# Models/Main function
#==============================================================================
import numpy as np
from libraries import layers
from libraries import activations
from libraries import losses

class Sequential(object):
    def __init__(self):
        # RunTime
        self.inputs = [];
        self.targets = [];
        self.layer_inputs = [];
        self.layer_activation_inputs = [];
        self.learning_rate = 0.001;
        self.loss_values = [];
        # Initializations
        self.layer_type, self.activation_type = [], [];
        self.layer_output_shapes = [];
        self.trainable_params_bool, self.layer_weights, self.layer_biases = [], [], [];
        # Loss and optimizers
        self.loss_type = [];
        self.optimizer_type = [];
        return
    def add(self, *positional_args):
        layer_type = positional_args[0];
        if self.layer_output_shapes==[]: #First layer
            input_shape, output_shape, weights, biases, trainable_params, activation_type = \
                getattr(layers, layer_type)(None, positional_args);
        else:
            input_shape, output_shape, weights, biases, trainable_params, activation_type = \
                getattr(layers, layer_type)(self.layer_output_shapes[-1], positional_args);
        self.layer_type.append(layer_type);
        self.activation_type.append(activation_type);
        self.layer_output_shapes.append(output_shape);
        self.layer_weights.append(weights);
        self.layer_biases.append(biases);
        self.trainable_params_bool.append(trainable_params);
        return
    def compile(self, **keyword_args):
        self.loss_type = keyword_args['loss'];
        self.optimizer_type = keyword_args['optimizer']
        return
    def train(self, inputs, targets, n_epochs=10):
        self.inputs = inputs;
        self.targets = targets;
        batch_size = 1;
        for epoch in range(n_epochs):
            print("Current Epoch: ", epoch);
            thisEpochLoss = 0;
            thisEpochIters = 0;
            batch_start = 0;
            while(batch_start+batch_size<=inputs.shape[0]):
                thisEpochIters+=1;
#                untilThis = int(10*batch_start/inputs.shape[0]);
#                print(">"*untilThis+"-"*(10-untilThis))
                for ind in range(batch_start, batch_start+batch_size):
                    # Forward Pass
                    currInput = inputs[ind];
                    currTarget = targets[ind];
                    for layer_ind, layer in enumerate(self.layer_type):
                        # Multiply weights and add biases
                        if self.trainable_params_bool[layer_ind]==True:
                            self.layer_inputs.append(currInput);
                            currInput = np.dot(currInput.T,self.layer_weights[layer_ind]);
                            currInput = currInput.T;
                            currInput = currInput + self.layer_biases[layer_ind];
                        else:
                            self.layer_inputs.append(None);
                        # If any activation, apply activation function
                        if self.activation_type[layer_ind]!=None:
                            self.layer_activation_inputs.append(currInput)
                            act_type = self.activation_type[layer_ind];
                            currInput = getattr(activations, act_type, False)(currInput);
                        else:
                            self.layer_activation_inputs.append(None)
                    error = getattr(losses, self.loss_type)(currInput, currTarget, False)
                    thisEpochLoss+=error;
                    #print(error) # currInput is the output on which loss function has to be applied
                    # Backward Pass
                    currLoss = getattr(losses, self.loss_type)(currInput, currTarget, True)
                    for layer_ind in range(len(self.layer_type)-1, 0, -1): #Input layer need not be updated!
                        if self.activation_type[layer_ind]!=None:
                            act_type = self.activation_type[layer_ind];
                            currLoss *= getattr(activations, act_type, True)(self.layer_activation_inputs[layer_ind]);
                        # Update Weights and biases
                        if self.trainable_params_bool[layer_ind]==True:
                            for i in range(self.layer_weights[layer_ind].shape[0]):
                                for j in range(self.layer_weights[layer_ind].shape[1]):
                                    self.layer_weights[layer_ind][i,j] -= self.learning_rate*self.layer_inputs[layer_ind][i]*currLoss[j];
                            for j in range(self.layer_weights[layer_ind].shape[1]):
                                self.layer_biases[layer_ind][j] -= self.learning_rate*currLoss[j];
                            # Pass on the currLoss
                            currLoss = np.dot(currLoss.T,self.layer_weights[layer_ind].T);
                            currLoss = currLoss.T;
                batch_start+=batch_size;
            print(thisEpochLoss/thisEpochIters);
        return









