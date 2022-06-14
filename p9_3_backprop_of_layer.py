# Backpropagation of layer of neurons
import numpy as np

dvalues = np.array([[1., 1., 1.]])  # 2D

weights = np.array([[0.2, 0.8, -0.5, 1],
                    [0.5, -0.91, 0.26, -0.5],
                    [-0.26, -0.27, 0.17, 0.87]]).T

dx0 = sum(weights[0]*dvalues[0])
dx1 = sum(weights[1]*dvalues[0])
dx2 = sum(weights[2]*dvalues[0])
dx3 = sum(weights[3]*dvalues[0])

dinputs = np.array([dx0, dx1, dx2, dx3])
#print(dinputs)

# DOT PRODUCT

dvalues = np.array([[1., 1., 1.]])  # 2D

weights = np.array([[0.2, 0.8, -0.5, 1],
                    [0.5, -0.91, 0.26, -0.5],
                    [-0.26, -0.27, 0.17, 0.87]])



dinputs = np.dot(dvalues[0], weights)
#print(dinputs)







# FOR BATCH OF SAMPLES, RELU ACTIVATION AND GETTING INPUTS, WEIGHTS AND BIASES (LENGTH OF 3)

# 3 sets of inputs - samples
inputs = np.array([[1., 2., 3., 2.5],
                   [2., 5., -1., 2],
                   [-1.5, 2.7, 3.3, -0.8]])

# Passed in gradient from the next layer
dvalues = np.array([[1., 1., 1.],
                    [2., 2., 2.],
                    [3., 3., 3.]])

# 3 sets of weights - one set for each neuron
# 4 inputs, thus 4 weights
# Recall that we are using the transpose
weights = np.array([[0.2, 0.8, -0.5, 1],            # features, neurons
                    [0.5, -0.91, 0.26, -0.5],
                    [-0.26, -0.27, 0.17, 0.87]]).T

# Biases are the row vector with shape (1, neurons)
biases = np.array([[2, 3, 0.5]])

# Forward step
layer_outputs = np.dot(inputs, weights) + biases # Dense Layer
relu_outputs = np.maximum(0, layer_outputs) # ReLU activation

# Backpropagation
# ReLU activation - simulates derivatives with respect to input values
# from next layer passed to current layer during backpropagation

# Samples, gradients times ReLU
drelu = dvalues.copy() # To not affect the original array and we are making a simplification
drelu[layer_outputs <= 0] = 0  # Simplification of ReLU derivative [1 if (x>0)] including dvalue (dvalue*drelu)

# Dense Layer
# dinputs - multiply by weights
dinputs = np.dot(drelu, weights.T)  # samples, features
# dweights - multiply by inputs
dweights = np.dot(inputs.T, drelu)  # features, neurons
# dbiases - sum values, do this over samples (first axis), keepdims
# since this by default will produce a plain list
dbiases = np.sum(drelu, axis=0, keepdims=True)

# Update parameters

weights += -0.001 * weights
biases += -0.001 * biases

print(weights)
print(biases)

