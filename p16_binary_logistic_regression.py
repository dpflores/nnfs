# Lets implement the binary logistic regression.
# This is an alternative output layer for neural network where each neuron
# represents two classes, 0 for one of the classes, and 1 for the other.
# This couldbe like cat vs dog or cat vs not cat


from pickletools import optimize
from xml.etree.ElementPath import prepare_descendant
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
dense1 = Layer_Dense(2, 64, weight_regularizer_l2=5e-4, bias_regularizer_l2=5e-4)

# Create ReLU activation (to be used with Dense layer):
activation1 = Activation_ReLU()

# Create seconde Dense layer with 64 input features (output of the previous layer here) and 1 output value
dense2 = Layer_Dense(64, 1)

# Create sigmoid activation
activation2 = Activation_Sigmoid()

# Create loss function
loss_function = Loss_BinaryCrossentropy()

# Create optimizer
optimizer = Optimizer_Adam(decay=5e-7)

# Train loop
for epoch in range(10001):
    # Forward pass
    # Perform a forward pass of our training data through this layer
    dense1.forward(X)

    # Perform a forward pass through activation function
    # takes the output of first dense layer here
    activation1.forward(dense1.output)

    # Perform a forward pass through second Dense layer
    # takes outputs of activation function of first layer as inputs
    dense2.forward(activation1.output)

    # Perform forward pass through activation function
    # takes the output of second layer here
    activation2.forward(dense2.output)

    # calculate data loss 
    data_loss = loss_function.calculate(activation2.output, y)

    # Calculate regularization penalty
    regularization_loss = loss_function.regularization_loss(dense1) + loss_function.regularization_loss(dense2)

    # Calculate overall loss
    loss = data_loss + regularization_loss

    # Calculate accuracy from output of activation2 and targets
    # Part in the brackets returns a binary mask - array consisting
    # of True/false values, multiplying it by 1 changes it into arrays of 1s and 0s

    predictions = (activation2.output>0.5)*1
    accuracy = np.mean(predictions==y)

    if not epoch % 100:
        print(f'epoch: {epoch}, ' +
              f'acc: {accuracy:.3f}, ' +
              f'loss: {loss:.3f}, (' +
              f'data_loss: {data_loss:.3f}, ' +
              f'reg_loss: {regularization_loss:.3f}), ' +
              f'lr: {optimizer.current_learning_rate}')

    # Backward pass
    loss_function.backward(activation2.output, y)
    dense2.backward(loss_function.dinputs)
    dense2.backward(activation2.dinputs)
    activation1.backward(dense1.dinputs)
    dense1.backward(activation1.dinputs)


    # Update weights and biases
    optimizer.pre_update_params()
    optimizer.update_params(dense1)
    optimizer.update_params(dense2)
    optimizer.post_update_params()



