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
        self.loss_type, self.output_loss_value = [], 'per_sample'; #total
        self.optimizer_type = [];
        self.batch_size = [];
        self.n_epochs, self.curr_epoch = [], [];
        self.verbose = []; 
        self.learning_rate = 0.0001;
        # Back propogation essentials
        self.layer_inputs, self.layer_activation_inputs = [], [];
        self.layer_delta_weights = [];
        self.layer_delta_biases = [];
        # Layer Attributes and initialize weights and biases
        self.layers = []; #sequential_layer_properties
        self.layer_weights= [];
        self.layer_biases = [];
        return
    def add(self, *positional_args, **keyword_args):
        # Add Known data for any layer
        thisLayer = (getattr(defaults, 'layer_defaults')).copy();
        if len(self.layers)>0:
            thisLayer['input_shape'] = self.layers[-1]['output_shape'];
        # Add collected data
        thisLayer['layer_type'] = positional_args[0];
        thisLayer, weights, biases = getattr(layers, thisLayer['layer_type'])(thisLayer, keyword_args);
        self.layers.append(thisLayer);
        self.layer_weights.append(weights);
        self.layer_biases.append(biases);
        return
    def compile(self, **keyword_args):
        try:
            assert(len(self.layers)==len(self.layer_weights)==len(self.layer_biases));
        except AssertionError:
            raise Exception('Layers Not properly added!PLease check source code')
        try:
            self.loss_type = keyword_args['loss'];
        except KeyError:
            raise Exception('loss type not defined!')
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
                    for layer_ind in range(len(self.layers)):
                        currInput = self.forwardPass(currInput, layer_ind);                       
                    # Calculate and Back-propogate Gradients wrt Error
                    error = (getattr(losses, self.loss_type)(currInput, currTarget, False)).astype(config.data_type);
                    batch_error+=error; 
                    currLoss = getattr(losses, self.loss_type)(currInput, currTarget, True)
                    # Collect delta values to be updated
                    for layer_ind in range(len(self.layers)-1, 0, -1): #Input layer need not be updated!
                        if self.layers[layer_ind]['any_params'] and self.layers[layer_ind]['trainable']:
                            currLoss = self.update_delta_weights_biases(currLoss, layer_ind);
                        elif self.layers[layer_ind]['layer_type']=='flatten_layer':
                            currLoss = self.update_delta_weights_biases(currLoss, layer_ind);
                # Update weights and biases with collected delta values for this batch
                for layer_ind in range(len(self.layers)-1, 0, -1): #Input layer need not be updated!
                    if self.layers[layer_ind]['any_params'] and self.layers[layer_ind]['trainable']:
                        self.update_weights_biases(layer_ind);
                # GO TO NEXT BATCH:
                batch_start+=self.batch_size;
                if self.output_loss_value=='per_sample':
                    self.error+=(batch_error/self.batch_size);
                elif self.output_loss_value=='total':
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
        if self.layers[layer_ind]['layer_type']=='dense_layer':
            #Check if there is dropout
            if self.layers[layer_ind]['dropout']!=0:
                currInput*=((np.random.rand(*currInput.shape)<self.layers[layer_ind]['dropout'])/self.layers[layer_ind]['dropout']); 
            # Multiply weights and add biases
            self.layer_inputs[layer_ind] = currInput.copy();
            currInput = np.dot(currInput.T,self.layer_weights[layer_ind]);
            currInput = currInput.T;
            currInput = currInput + self.layer_biases[layer_ind];
            # If any activation, apply activation function
            if self.layers[layer_ind]['activation_type']!=None:
                self.layer_activation_inputs[layer_ind] = currInput.copy();
                act_type = self.layers[layer_ind]['activation_type'];
                currInput = getattr(activations, act_type, False)(currInput);
        # Convolution Layer Type
        elif self.layers[layer_ind]['layer_type']=='conv2D_layer':
            #Check if there is dropout
            if self.layers[layer_ind]['dropout']!=0:
                currInput*=((np.random.rand(*currInput.shape)<self.layers[layer_ind]['dropout'])/self.layers[layer_ind]['dropout']); 
            # Set sizes for convolution
            kernel_height, kernel_width = self.layers[layer_ind]['kernel_shape']
            stride_height, stride_width = self.layers[layer_ind]['stride']
            if self.layers[layer_ind]['padding']=='same': # 'same' size padding type
                currInput = getattr(helpers, 'padder')(currInput, kernel_height-1, kernel_width-1); # Stride will be (1,1)
            # Multiply weights and add biases
            self.layer_inputs[layer_ind] = currInput.copy();
            conv_maps = np.zeros(self.layers[layer_ind]['output_shape'],dtype=config.data_type)
            for kernel_ind in range(self.layers[layer_ind]['n_kernels']):
                i, j = 0, 0;
                while i<self.layers[layer_ind]['output_shape'][0]: #range(currInput.shape[0]-kernel_height+1):
                    while j<self.layers[layer_ind]['output_shape'][1]: #range(currInput.shape[1]-kernel_width+1):
                        for m in range(0,kernel_height):
                            for n in range(0,kernel_width):
                                conv_maps[int(i/stride_height),int(j/stride_width),kernel_ind]+=(np.sum(currInput[i+m,j+n,:]*self.layer_weights[layer_ind][kernel_ind,m,n,:]));
                        j+=stride_width;
                    i+=stride_height;
                conv_maps[:,:,kernel_ind]+=self.layer_biases[layer_ind][kernel_ind];
            currInput = conv_maps;
            # If any activation, apply activation function
            if self.layers[layer_ind]['activation_type']!=None:
                self.layer_activation_inputs[layer_ind] = currInput.copy();
                act_type = self.layers[layer_ind]['activation_type'];
                currInput = getattr(activations, act_type, False)(currInput);
        # Flatten Layer Type
        elif self.layers[layer_ind]['layer_type']=='flatten_layer':
            currInput = getattr(helpers, 'flatten')(currInput, currInput.shape);
        # Return updated currInput
        return currInput;
    def initialize_delta_weights_biases(self):
        self.layer_delta_weights, self.layer_delta_biases = [], [];
        for layer_ind in range(len(self.layers)):
            if self.layers[layer_ind]['any_params'] and self.layers[layer_ind]['trainable']:
                self.layer_delta_weights.append(np.zeros(self.layer_weights[layer_ind].shape,dtype=config.data_type));
                self.layer_delta_biases.append(np.zeros(self.layer_biases[layer_ind].shape,dtype=config.data_type));
            else:
                self.layer_delta_weights.append(None);
                self.layer_delta_biases.append(None);
        return
    def update_delta_weights_biases(self, currLoss, layer_ind):
        # Dense Layer Type
        if self.layers[layer_ind]['layer_type']=='dense_layer':
            # Pass loss through activation if any
            if self.layers[layer_ind]['activation_type']!=None: 
                act_type = self.layers[layer_ind]['activation_type'];
                currLoss *= getattr(activations, act_type, True)(self.layer_activation_inputs[layer_ind]);
            if self.layers[layer_ind]['trainable']:
                # Collecting Delta values for this training sample (ONE sample) in this batch
                for i in range(self.layer_weights[layer_ind].shape[0]):
                    for j in range(self.layer_weights[layer_ind].shape[1]):
                        self.layer_delta_weights[layer_ind][i,j] += ((-1)*self.learning_rate*self.layer_inputs[layer_ind][i]*currLoss[j]);
                for j in range(self.layer_weights[layer_ind].shape[1]):
                    self.layer_delta_biases[layer_ind][j] += ((-1)*self.learning_rate*currLoss[j]);
            # Pass on the currLoss
            currLoss = np.dot(currLoss.T,self.layer_weights[layer_ind].T);
            currLoss = currLoss.T;
        # Convolution Layer Types
        elif self.layers[layer_ind]['layer_type']=='conv2D_layer':
            # Pass loss through activation if any
            if self.layers[layer_ind]['activation_type']!=None: 
                act_type = self.layers[layer_ind]['activation_type'];
                currLoss *= getattr(activations, act_type, True)(self.layer_activation_inputs[layer_ind]);
            kernel_height, kernel_width = self.layers[layer_ind]['kernel_shape']
            output_height, output_width, output_channels = self.layers[layer_ind]['output_shape']
            stride_height, stride_width = self.layers[layer_ind]['stride']
            input_height, input_width, input_channels = self.layers[layer_ind]['input_shape']
            if self.layers[layer_ind]['trainable']:
                # Collecting Delta values for this training sample (ONE sample) in this batch
                for kernel_ind in range(self.layers[layer_ind]['n_kernels']):
                    for channel in range(self.layers[layer_ind]['input_shape'][-1]):
                        for i in range(kernel_height):
                            for j in range(kernel_width):
                                self.layer_delta_weights[layer_ind][kernel_ind,i,j,channel] += ((-1)*self.learning_rate*np.sum(currLoss[:,:,kernel_ind]*self.layer_inputs[layer_ind][np.arange(i,i+output_height*stride_height,stride_height),np.arange(j,j+output_width*stride_width,stride_width),channel]));
                for kernel_ind in range(self.layers[layer_ind]['n_kernels']):                    
                    self.layer_delta_biases[layer_ind][kernel_ind]+=((-1)*self.learning_rate*np.sum(currLoss[:,:,kernel_ind]));
            # Pass on the currLoss
#            tempLoss = np.zeros(self.layers[layer_ind]['input_shape'], dtype=config.data_type)
#            input_height, input_width, input_channels = self.layers[layer_ind]['input_shape']
#            for channel_ind in range(input_channels):
#                for i in range(input_height):
#                    for j in range(input_width):
#                        for m in range(i-(kernel_height-1),i+1):
#                            for n in range(j-(kernel_width-1),j+1):
#                                for kernel_ind in range(self.layers[layer_ind]['n_kernels']):
#                                    if m>=0 and n>=0 and m<input_height and n<input_width:
#                                        tempLoss[i,j,channel_ind]+=(currLoss[m,n,kernel_ind]*self.layer_weights[layer_ind][kernel_ind,i-m,j-n,channel_ind]);
            tempLoss = np.zeros(self.layers[layer_ind]['input_shape'], dtype=config.data_type)
            for channel_ind in range(input_channels):
                for i in range(input_height):
                    for j in range(input_width):
                        for m in range(i-(kernel_height-1),i+1):
                            for n in range(j-(kernel_width-1),j+1):
                                for kernel_ind in range(self.layers[layer_ind]['n_kernels']):
                                    if m>=0 and n>=0 and m<=((output_height-1)*stride_height) and n<=((output_width-1)*stride_width) \
                                        and m%stride_height==0 and n%stride_width==0:
                                           tempLoss[i,j,channel_ind]+=(currLoss[int(m/stride_height),int(n/stride_width),kernel_ind]*self.layer_weights[layer_ind][kernel_ind,i-m,j-n,channel_ind]);
            currLoss = tempLoss;
        # Flatten layer Type
        elif self.layers[layer_ind]['layer_type']=='flatten_layer':
            currLoss = getattr(helpers, 'unflatten')(currLoss, self.layers[layer_ind]['input_shape']);
        # Return the Loss Updated
        return currLoss;
    def update_weights_biases(self, layer_ind): # FOR THIS BATCH: Update Weights and biases
        # Dense Layer Type
        if self.layers[layer_ind]['layer_type']=='dense_layer' and self.layers[layer_ind]['trainable']:
            for i in range(self.layer_weights[layer_ind].shape[0]):
                for j in range(self.layer_weights[layer_ind].shape[1]):
                    self.layer_weights[layer_ind][i,j] += self.layer_delta_weights[layer_ind][i,j]
            for j in range(self.layer_weights[layer_ind].shape[1]):
                self.layer_biases[layer_ind][j] += self.layer_delta_biases[layer_ind][j];
            return
        # Convolution Layer Types
        elif self.layers[layer_ind]['layer_type']=='conv2D_layer' and self.layers[layer_ind]['trainable']:
            kernel_height, kernel_width = self.layers[layer_ind]['kernel_shape']
            input_height, input_width, input_channels = self.layers[layer_ind]['input_shape'];
            for kernel_ind in range(self.layers[layer_ind]['n_kernels']):
                for channel in range(self.layers[layer_ind]['input_shape'][-1]):
                    for i in range(kernel_height):
                        for j in range(kernel_width):
                            self.layer_weights[layer_ind][kernel_ind,i,j,channel] += self.layer_delta_weights[layer_ind][kernel_ind,i,j,channel]
            for kernel_ind in range(self.layers[layer_ind]['n_kernels']):                    
                self.layer_biases[layer_ind][kernel_ind]+=self.layer_delta_biases[layer_ind][kernel_ind]
            return
