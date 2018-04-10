# -*- coding: utf-8 -*-
"""
Created on Tue Apr 10 17:21:58 2018
@author: s.jayanthi
"""
from libraries.models import Sequential

model = Sequential();
model.add('input_layer', 10);
model.add('dense_layer', 32);
model.add('activation_layer', 'relu');
model.add('dense_layer', 64);
model.add('activation_layer', 'relu');
model.add('dense_layer', 32);
model.add('activation_layer', 'relu');
model.add('output_layer', 20);
model.compile('softmax_with_cross_entropy');