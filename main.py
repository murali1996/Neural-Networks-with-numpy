#==============================================================================
# -*- coding: utf-8 -*-
# Created on Tue Apr 10 17:21:58 2018
# @author: s.jayanthi
# Activation Function and their derivatives
#==============================================================================
import numpy as np

# =============================================================================
# from libraries.models import Sequential
# inputs = np.random.randn(250).reshape([25,10,1])
# targets = np.random.randint(0,2,(25,5,1))
# model = Sequential();
# model.add('input_layer', shape=(10,));
# model.add('dense_layer', shape=(32,), activation_type='relu');
# model.add('dense_layer', shape=(64,), activation_type='relu');
# model.add('dense_layer', shape=(128,), activation_type='relu');
# model.add('dense_layer', shape=(32,), activation_type='relu');
# # Softmax with cross-entropy
# #model.add('dense_layer', shape=(5,));
# #model.compile(loss = 'softmax_with_cross_entropy', optimizer='gradient_descent');
# # Sigmoid with binary cross entropy
# #model.add('dense_layer', shape=(5,), activation_type='sigmoid');
# #model.compile(loss = 'binary_cross_entropy', optimizer='gradient_descent');
# # Sigmoid with rmse
# #model.add('dense_layer', shape=(1,), activation_type='sigmoid');
# #model.compile(loss = 'rmse', optimizer='gradient_descent');
# model.train(inputs=inputs,targets=targets, n_epochs=100, verbose=0, batch_size=1)
# =============================================================================

from libraries.models import Sequential
inputs = np.random.randn(30720).reshape([10,32,32,3])
targets = np.random.randint(0,2,(10,32,32,1))
model = Sequential();
model.add('input_layer', shape=(32,32,3));
model.add('conv2D_layer', kernel_shape=(5,5), n_kernels=10, stride=(1,1), 
          activation_type='relu', padding='same'); 
model.add('conv2D_layer', kernel_shape=(3,3), n_kernels=15, stride=(1,1), 
          activation_type='relu', padding='same'); 
# Reduce to Dense Layer 
#model.add('conv2D_layer', kernel_shape=(3,3), n_kernels=1, stride=(1,1), 
#          activation_type='relu', padding='same');
#model.add('flatten_layer');
#model.add('dense_layer', shape=(5,), activation_type='relu');
#model.compile(loss = 'softmax_with_cross_entropy', optimizer='gradient_descent');
# Keep the output as an image
model.add('conv2D_layer', kernel_shape=(3,3), n_kernels=1, stride=(1,1), 
          activation_type='sigmoid', padding='same');
model.compile(loss = 'binary_cross_entropy', optimizer='gradient_descent');
model.train(inputs=inputs,targets=targets, n_epochs=100)

