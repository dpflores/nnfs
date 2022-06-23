# Softmax activation

import math
import numpy as np


layer_outputs = [[4.8, 1.21, 2.385],  # neurons * sample
                 [8.9, -1.81, 0.2],
                 [1.41, 1.051, 0.026]]

layer_outputs = layer_outputs - np.max(layer_outputs, axis=1, keepdims=True) # This is to avoid overflow because the max will be 0

exp_values = np.exp(layer_outputs)
norm_values = exp_values / np.sum(exp_values, axis=1, keepdims=True) #adding the rows using axi=1 but as column


print(norm_values)

