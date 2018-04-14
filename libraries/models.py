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
from libraries import defaults

class Sequential(object):
    def __init__(self):
        # Forward Propogation Essentials
        self.inputs, self.targets = [], [];
        self.error, self.val_error = [], [];
        # Convex Optimization/ Training Essentials     
        self.loss_type, self.optimizer_type = [], [];
        self.batch_size, self.n_epochs, self.curr_epoch, self.verbose = [], [], [], []; 
        self.learning_rate = 0.002;
        # Back propogation essentials
        self.layer_inputs, self.layer_activation_inputs = [], [];
        self.layer_delta_weights, self.layer_delta_biases = [], [];
        # Layer Attributes and initialize weights and biases
        self.layer_properties = [];
        self.layer_weights, self.layer_biases = [], [];
        return
    def add(self, *positional_args, **keyword_args):
        # Add Known data for any layer
        thisLayer = (getattr(defaults, 'layer_defaults')).copy();
        if len(self.layer_properties)>0:
            thisLayer['input_shape'] = self.layer_properties[-1]['output_shape'];
        # Add collected data
        thisLayer['layer_type'] = positional_args[0];
        thisLayer, weights, biases = getattr(layers, thisLayer['layer_type'])(thisLayer, keyword_args);
        self.layer_properties.append(thisLayer);
        self.layer_weights.append(weights);
        self.layer_biases.append(biases);
        return
    def compile(self, **keyword_args):
        try:
            assert(len(self.layer_properties)==len(self.layer_weights)==len(self.layer_biases));
        except AssertionError:
            raise Exception('Layers Not properly added!PLease check source code')
        self.loss_type = keyword_args['loss'];
        self.optimizer_type = keyword_args['optimizer']; # TIP: Not used as of now!!
        return
    def train(self, inputs, targets, batch_size=1, n_epochs=1, verbose=0):
        self.inputs = inputs.astype(config.data_type);
        self.targets = targets.astype(config.data_type);
        self.batch_size = batch_size;
        self.n_epochs = n_epochs;
        self.verbose = verbose;
        self.curr_epoch = 1;
        n_batches = int(self.inputs.shape[0]/self.batch_size);
        while self.curr_epoch<=self.n_epochs:
            # Print Epoch Number
            print("Epoch: {}/{}".format(self.curr_epoch, self.n_epochs),end='\n');
            self.error = np.zeros(1, dtype=config.data_type);
            batch_start = 0;
            while(batch_start+self.batch_size<=self.inputs.shape[0]):
                # FOR THIS BATCH: # Print Batch Number
                batch_error = np.zeros(1, dtype=config.data_type);
                nthBatch = int(batch_start/self.batch_size);
                to_be_printed = ('>'*nthBatch+'-'*(n_batches-nthBatch))
                print(to_be_printed, end="\r") if verbose else print('',end='\r');
                # FOR THIS BATCH: Initialize delta weights and delta biases
                self.initialize_delta_weights_biases();
                for ind in range(batch_start, batch_start+self.batch_size):
                    # Forward Pass
                    currInput, currTarget = inputs[ind], targets[ind]; 
                    for layer_ind in range(len(self.layer_properties)):
                        currInput = self.forwardPass(currInput, layer_ind);                       
                    # Calculate and Back-propogate Gradients wrt Error
                    error = (getattr(losses, self.loss_type)(currInput, currTarget, False)).astype(config.data_type);
                    batch_error+=error; 
                    currLoss = getattr(losses, self.loss_type)(currInput, currTarget, True)
                    # Collect delta values to be updated
                    for layer_ind in range(len(self.layer_properties)-1, 0, -1): #Input layer need not be updated!
                        if self.layer_properties[layer_ind]['any_params'] and self.layer_properties[layer_ind]['trainable']:
                            currLoss = self.update_delta_weights_biases(currLoss, layer_ind);
                # Update weights and biases with collected delta values for this batch
                for layer_ind in range(len(self.layer_properties)-1, 0, -1): #Input layer need not be updated!
                    if self.layer_properties[layer_ind]['any_params'] and self.layer_properties[layer_ind]['trainable']:
                        self.update_weights_biases(layer_ind);
                # GO TO NEXT BATCH:
                batch_start+=self.batch_size;
                self.error+=batch_error;
            # GOTO next EPOCH
            print('>'*n_batches, end='\n'); print(self.error);
            self.curr_epoch+=1;
        return
    def forwardPass(self, currInput, layer_ind):
        # Initialization
        self.layer_inputs.append(None);
        self.layer_activation_inputs.append(None);
        # Dense Layer Type
        if self.layer_properties[layer_ind]['layer_type']=='dense_layer':
            # Multiply weights and add biases
            self.layer_inputs[layer_ind] = currInput;
            currInput = np.dot(currInput.T,self.layer_weights[layer_ind]);
            currInput = currInput.T;
            currInput = currInput + self.layer_biases[layer_ind];
            # If any activation, apply activation function
            if self.layer_properties[layer_ind]['activation_type']!=None:
                self.layer_activation_inputs[layer_ind] = currInput;
                act_type = self.layer_properties[layer_ind]['activation_type'];
                currInput = getattr(activations, act_type, False)(currInput);
        # Convolution Layer Type
        elif self.layer_properties[layer_ind]['layer_type']=='conv2D_layer':
            print('Under Construction!')
        # Return updated currInput
        return currInput;
    def initialize_delta_weights_biases(self):
        self.layer_delta_weights, self.layer_delta_biases = [], [];
        for layer_ind in range(len(self.layer_properties)):
            if self.layer_properties[layer_ind]['any_params'] and self.layer_properties[layer_ind]['trainable']:
                self.layer_delta_weights.append(np.zeros(self.layer_weights[layer_ind].shape,dtype=config.data_type));
                self.layer_delta_biases.append(np.zeros(self.layer_biases[layer_ind].shape,dtype=config.data_type));
            else:
                self.layer_delta_weights.append(None);
                self.layer_delta_biases.append(None);
        return
    def update_delta_weights_biases(self, currLoss, layer_ind):
        # Dense Layer Type
        if self.layer_properties[layer_ind]['layer_type']=='dense_layer':
            # Pass loss through activation if any
            if self.layer_properties[layer_ind]['activation_type']!=None: 
                act_type = self.layer_properties[layer_ind]['activation_type'];
                currLoss *= getattr(activations, act_type, True)(self.layer_activation_inputs[layer_ind]);
            # Collecting Delta values for this one training sample in this batch
            for i in range(self.layer_weights[layer_ind].shape[0]):
                for j in range(self.layer_weights[layer_ind].shape[1]):
                    self.layer_delta_weights[layer_ind][i,j] += (-1)*self.learning_rate*self.layer_inputs[layer_ind][i]*currLoss[j];
            for j in range(self.layer_weights[layer_ind].shape[1]):
                self.layer_delta_biases[layer_ind][j] += (-1)*self.learning_rate*currLoss[j];
            # Pass on the currLoss
            currLoss = np.dot(currLoss.T,self.layer_weights[layer_ind].T);
            currLoss = currLoss.T;
        # Convolution Layer Types
        elif self.layer_properties[layer_ind]['layer_type']=='conv2D_layer':
            print('Under Construction!')
        # Return the Loss Updated
        return currLoss;
    def update_weights_biases(self, layer_ind): # FOR THIS BATCH: Update Weights and biases
        # Dense Layer Type
        if self.layer_properties[layer_ind]['layer_type']=='dense_layer':
            for i in range(self.layer_weights[layer_ind].shape[0]):
                for j in range(self.layer_weights[layer_ind].shape[1]):
                    self.layer_weights[layer_ind][i,j] += self.layer_delta_weights[layer_ind][i,j]
            for j in range(self.layer_weights[layer_ind].shape[1]):
                self.layer_biases[layer_ind][j] += self.layer_delta_biases[layer_ind][j];
            return
        # Convolution Layer Types
        elif self.layer_properties[layer_ind]['layer_type']=='conv2D_layer':
            print('Under Construction!')
            return
