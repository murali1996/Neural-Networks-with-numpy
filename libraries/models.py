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
from libraries import helpers

class Sequential(object):
    def __init__(self):
        # Forward Propogation Essentials
        self.inputs, self.targets = [], [];
        self.error, self.val_error = [], [];
        # Convex Optimization/ Training Essentials 
        self.model_compiled = False;
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
        self.model_compiled = True;
        return
    def train(self, inputs, targets, batch_size=1, n_epochs=1, verbose=0):
        if not self.model_compiled:
            raise Exception('Compile your model before training.')
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
                        elif self.layer_properties[layer_ind]['layer_type']=='flatten_layer':
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
            # Check if padding required
            if self.layer_properties[layer_ind]['padding']=='same':
                kernel_height, kernel_width = self.layer_properties[layer_ind]['kernel_shape']
                currInput = getattr(helpers, 'padder')(currInput, kernel_height-1, kernel_width-1);
            # Multiply weights and add biases
            self.layer_inputs[layer_ind] = currInput;
            conv_maps = np.zeros(self.layer_properties[layer_ind]['output_shape'],dtype=config.data_type)
            for kernel_ind in range(self.layer_properties[layer_ind]['n_kernels']):
                for i in range(currInput.shape[0]-kernel_height+1):
                    for j in range(currInput.shape[1]-kernel_width+1):
                        for m in range(i,i+kernel_height):
                            for n in range(j,j+kernel_width):
                                conv_maps[i,j,kernel_ind]+=np.sum(currInput[m,n,:]*self.layer_weights[layer_ind][kernel_ind,m-i,n-j,:]);
                conv_maps[:,:,kernel_ind]+=self.layer_biases[layer_ind][kernel_ind];
            currInput = conv_maps;
            # If any activation, apply activation function
            if self.layer_properties[layer_ind]['activation_type']!=None:
                self.layer_activation_inputs[layer_ind] = currInput;
                act_type = self.layer_properties[layer_ind]['activation_type'];
                currInput = getattr(activations, act_type, False)(currInput);
        # Flatten Layer Type
        elif self.layer_properties[layer_ind]['layer_type']=='flatten_layer':
            currInput = getattr(helpers, 'flatten')(currInput, currInput.shape);
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
            # Pass loss through activation if any
            if self.layer_properties[layer_ind]['activation_type']!=None: 
                act_type = self.layer_properties[layer_ind]['activation_type'];
                currLoss *= getattr(activations, act_type, True)(self.layer_activation_inputs[layer_ind]);
            # Collecting Delta values for this one training sample in this batch
            kernel_height, kernel_width = self.layer_properties[layer_ind]['kernel_shape']
            input_height, input_width, input_channels = self.layer_properties[layer_ind]['input_shape'];
            for kernel_ind in range(self.layer_properties[layer_ind]['n_kernels']):
                for channel in range(self.layer_properties[layer_ind]['input_shape'][-1]):
                    for i in range(kernel_height):
                        for j in range(kernel_width):
                            self.layer_delta_weights[layer_ind][kernel_ind,i,j,channel] += (-1)*self.learning_rate*\
                                np.sum(self.layer_inputs[layer_ind][i:i+input_height,j:j+input_width,channel]*currLoss[0:0+input_height,0:0+input_width,kernel_ind]);
            for kernel_ind in range(self.layer_properties[layer_ind]['n_kernels']):                    
                self.layer_delta_biases[layer_ind][kernel_ind]+= (-1)*self.learning_rate*np.sum(currLoss[:,:,kernel_ind]);
            # Pass on the currLoss
            tempLoss = np.zeros(self.layer_properties[layer_ind]['input_shape'], dtype=config.data_type)
            for channel_ind in range(input_channels):
                for i in range(input_height):
                    for j in range(input_width):
                        for m in range(i-(kernel_height-1),i+1):
                            for n in range(j-(kernel_width-1),j+1):
                                for kernel_ind in range(self.layer_properties[layer_ind]['n_kernels']):
                                    if m>=0 and n>=0 and m<input_height and n<input_width:
                                        tempLoss[i,j,channel_ind]+=(currLoss[m,n,kernel_ind]*self.layer_weights[layer_ind][kernel_ind,i-m,j-n,channel_ind]);
            currLoss = tempLoss;
        # Flatten layer Type
        elif self.layer_properties[layer_ind]['layer_type']=='flatten_layer':
            currLoss = getattr(helpers, 'unflatten')(currLoss, self.layer_properties[layer_ind]['input_shape']);
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
            kernel_height, kernel_width = self.layer_properties[layer_ind]['kernel_shape']
            input_height, input_width, input_channels = self.layer_properties[layer_ind]['input_shape'];
            for kernel_ind in range(self.layer_properties[layer_ind]['n_kernels']):
                for channel in range(self.layer_properties[layer_ind]['input_shape'][-1]):
                    for i in range(kernel_height):
                        for j in range(kernel_width):
                            self.layer_weights[layer_ind][kernel_ind,i,j,channel] += self.layer_delta_weights[layer_ind][kernel_ind,i,j,channel]
            for kernel_ind in range(self.layer_properties[layer_ind]['n_kernels']):                    
                self.layer_biases[layer_ind][kernel_ind]+=self.layer_delta_biases[layer_ind][kernel_ind]
            return
