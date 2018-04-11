# -*- coding: utf-8 -*-
"""
Created on Tue Apr 10 17:21:58 2018
@author: s.jayanthi
"""
import numpy as np

from libraries.models import Sequential
inputs = np.random.randn(100).reshape([10,10,1])
targets = np.random.randint(0,2,(10,5,1))
model = Sequential();
model.add('input_layer', 10);
model.add('dense_layer', 32, 'relu');
model.add('dense_layer', 64, 'relu');
model.add('dense_layer', 32, 'relu');
model.add('output_layer', 5);
model.compile(loss = 'softmax_with_cross_entropy', optimizer='gradient_descent');
model.train(inputs=inputs,targets=targets, n_epochs=100) #batch_size=16