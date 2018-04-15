#==============================================================================
# -*- coding: utf-8 -*-
# Created on Sun Apr 15 03:47:53 2018
# @author: s.jayanthi
# Activation Function and their derivatives
#==============================================================================

layer_defaults = {'layer_type':None,
                  'input_shape':None,
                  'output_shape':None,
                  'kernel_shape':(3,3,None),
                  'n_kernels':None,
                  'stride':(1,1), # Specified along (height, width)
                  'padding':'same',
                  'activation_type':None,
                  'any_params':False,
                  'trainable':False,
                  }