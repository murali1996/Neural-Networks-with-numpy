#==============================================================================
# -*- coding: utf-8 -*-
# Created on Tue Apr 10 14:57:54 2018
# @author: s.jayanthi
# Models/Main function
#==============================================================================

from libraries import layers
from libraries import activations

class Sequential(object):
    def __init__(self):
        # RunTime
        self.input = [];
        self.target = [];
        self.layer_inputs = [];
        # Initializations
        self.layers = [];
        self.layer_weights = [];
        self.layer_output_shapes = [];
        return
    def add(self, *positional_args):
        layer_type = positional_args[0];
        self.layers.append(layer_type);
        input_shape, output_shape, weights = getattr(layers, layer_type)(positional_args[1]);
        self.layer_weights.append(weights);
        self.layer_output_shapes.append(output_shape);
        return
    def train(self):
        return