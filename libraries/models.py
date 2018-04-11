#==============================================================================
# -*- coding: utf-8 -*-
# Created on Tue Apr 10 14:57:54 2018
# @author: s.jayanthi
# Models/Main function
#==============================================================================
import numpy as np
from libraries import config
from libraries import layers
from libraries import activations
from libraries import losses

class Sequential(object):
    def __init__(self):
        # RunTime- Forward pass
        self.inputs, self.targets = [], [];
        self.layer_inputs, self.layer_activation_inputs = [], [];
        # RunTime- Backward pass
        self.learning_rate = 0.002;
        self.layer_delta_weights, self.layer_delta_biases = [], [];
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
        self.trainable_params_bool.append(trainable_params);
        self.layer_weights.append(weights);
        self.layer_biases.append(biases);
        return
    def compile(self, **keyword_args):
        self.loss_type = keyword_args['loss'];
        self.optimizer_type = keyword_args['optimizer']
        return
    def train(self, inputs, targets, n_epochs=10, verbose=0):
        self.inputs = inputs.astype(config.data_type);
        self.targets = targets.astype(config.data_type);
        batch_size = 2;
        for epoch in range(n_epochs):
            print("Epoch: {}/{}".format(epoch,n_epochs),end='\n');
            # Initializations
            thisEpochLoss = np.zeros(1, dtype=config.data_type);
            thisEpoch_nBatches = int(inputs.shape[0]/batch_size);
            batch_start = 0;
            while(batch_start+batch_size<=inputs.shape[0]):
                # Print Progress
                nthBatch = int(batch_start/batch_size)+1;
                if verbose:
                    print('>'*nthBatch+'-'*(thisEpoch_nBatches-nthBatch), end='\r');
                # FOR THIS BATCH: Initialize delta weights and delta biases
                self.layer_delta_weights, self.layer_delta_biases = [], [];
                for layer_ind in range(len(self.trainable_params_bool)):
                    if self.trainable_params_bool[layer_ind]==True:
                        self.layer_delta_weights.append(np.zeros(self.layer_weights[layer_ind].shape,dtype=config.data_type));
                        self.layer_delta_biases.append(np.zeros(self.layer_biases[layer_ind].shape,dtype=config.data_type));
                    else:
                        self.layer_delta_weights.append(None);
                        self.layer_delta_biases.append(None);
                # FOR THIS BATCH: Forward and backward pass
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
                    error = error.astype(config.data_type);
                    thisEpochLoss+=error; #print(error) # currInput is the output on which loss function has to be applied
                    # Backward Pass # Collect delta values to be updated
                    currLoss = getattr(losses, self.loss_type)(currInput, currTarget, True)
                    for layer_ind in range(len(self.layer_type)-1, 0, -1): #Input layer need not be updated!
                        if self.activation_type[layer_ind]!=None: # Pass loss through activation if any
                            act_type = self.activation_type[layer_ind];
                            currLoss *= getattr(activations, act_type, True)(self.layer_activation_inputs[layer_ind]);
                        if self.trainable_params_bool[layer_ind]==True:  # Delta values collection for this one training sample in this batch
                            for i in range(self.layer_weights[layer_ind].shape[0]):
                                for j in range(self.layer_weights[layer_ind].shape[1]):
                                    self.layer_delta_weights[layer_ind][i,j] += (-1)*self.learning_rate*self.layer_inputs[layer_ind][i]*currLoss[j];
                            for j in range(self.layer_weights[layer_ind].shape[1]):
                                self.layer_delta_biases[layer_ind][j] += (-1)*self.learning_rate*currLoss[j];
                            # Pass on the currLoss
                            currLoss = np.dot(currLoss.T,self.layer_weights[layer_ind].T);
                            currLoss = currLoss.T;
                # FOR THIS BATCH: Update Weights and biases
                for layer_ind in range(len(self.layer_type)-1, 0, -1): #Input layer need not be updated!
                    if self.trainable_params_bool[layer_ind]==True:
                        for i in range(self.layer_weights[layer_ind].shape[0]):
                            for j in range(self.layer_weights[layer_ind].shape[1]):
                                self.layer_weights[layer_ind][i,j] += self.layer_delta_weights[layer_ind][i,j]
                        for j in range(self.layer_weights[layer_ind].shape[1]):
                            self.layer_biases[layer_ind][j] += self.layer_delta_biases[layer_ind][j];
                # GO TO NEXT BATCH:
                batch_start+=batch_size;
            print('>'*thisEpoch_nBatches, end='\n');
            print(thisEpochLoss/thisEpoch_nBatches);
        return









