# Lets implement the binary logistic regression.
# This is an alternative output layer for neural network where each neuron
# represents two classes, 0 for one of the classes, and 1 for the other.
# This couldbe like cat vs dog or cat vs not cat


from mynnfs import * # My neural network

from mydatasets import spiral_data

import numpy as np

np.random.seed(0) # Same initialization 

# Create datasets of two classes
X,y = spiral_data(samples=100, classes = 2) # two internal features

# Reshape labels since they are not sparse anymore
# list of lists that contain one output (0 or 1)
# per output neuron, 1 neuron in this case
y = y.reshape(-1, 1)

# Create Dense layer with 2 input features and 64 output values (64 neurons)
dense1 = Layer_Dense(2, 512, weight_regularizer_l2=5e-4, bias_regularizer_l2=5e-4)

# Create ReLU activation (to be used with Dense layer):
activation1 = Activation_ReLU()

# Create seconde Dense layer with 64 input features (output of the previous layer here) and 1 output value
dense2 = Layer_Dense(64, 1)

# Create sigmoid activation
activation2 = Activation_Sigmoid


