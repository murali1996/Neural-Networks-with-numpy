#==============================================================================
# -*- coding: utf-8 -*-
# Created on Tue Apr 10 17:21:58 2018
# @author: s.jayanthi
# Activation Function and their derivatives
#==============================================================================
import numpy as np

#from libraries.models import Sequential
#inputs = np.random.randn(250).reshape([25,10,1])
#targets = np.random.randint(0,2,(250,5,1))
#model = Sequential();
#model.add('input_layer', shape=(10,));
#model.add('dense_layer', shape=(32,), activation_type='relu');
#model.add('dense_layer', shape=(64,), activation_type='relu');
#model.add('dense_layer', shape=(128,), activation_type='relu');
#model.add('dense_layer', shape=(32,), activation_type='relu');
#model.add('dense_layer', shape=(5,));
#model.compile(loss = 'softmax_with_cross_entropy', optimizer='gradient_descent');
#model.train(inputs=inputs,targets=targets, n_epochs=100, verbose=0) #batch_size=16

from libraries.models import Sequential
model = Sequential();
model.add('input_layer', shape=(128,128,3));
model.add('conv2D_layer', kernel_shape=(5,5,3), n_kernels=10, stride=(1,1), 
          activation_type='relu', padding='same'); 
model.add('conv2D_layer', kernel_shape=(5,5,3), n_kernels=15, stride=(1,1), 
          activation_type='relu', padding='same'); 
model.add('conv2D_layer', kernel_shape=(5,5,3), n_kernels=1, stride=(1,1), 
          activation_type='relu', padding='same'); 
#model.add('dense_layer', (32,), 'relu');
#model.add('dense_layer', (64,), 'relu');
#model.add('dense_layer', (128,), 'relu');
#model.add('dense_layer', (32,), 'relu');
#model.add('dense_layer', (5,));
#model.compile(loss = 'softmax_with_cross_entropy', optimizer='gradient_descent');
#model.train(inputs=inputs,targets=targets, n_epochs=100) #batch_size=16

