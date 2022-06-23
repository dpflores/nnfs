# We know how to classify data, but what if we want to predict the output of a data
# That is why we implement regression
# One problem with regression is that there is no a best way to get the accuracy
# Although it is better to have a good metric for accuracy

from cmath import sin
from mynnfs import * # My neural network

from mydatasets import sine_data

import numpy as np

np.random.seed(0) # Same initialization 

# Create dataset
X,y = sine_data()

# Create Dense Layer with 1 input feature and 64 output values
dense1 = Layer_Dense(1, 64)

# Create ReLU activation (to be used with dense layer)
activation1 = Activation_ReLU()

# Create second dense layer with 64 input features (from the previous layer)
# and 64 output value
dense2 = Layer_Dense(64,64)

# Create ReLU activation (to be used with dense layer)
activation2 = Activation_ReLU()

# Create third Dense Layer with 64 input features and 1 output value
dense3 = Layer_Dense(64,1)

# Create linear activation
activation3 = Activation_Linear()

# Create loss function
loss_function = Loss_MeanSquaredError()

# Create Optimizer 
optimizer = Optimizer_Adam(learning_rate=0.005, decay=1e-3)

# Accuracy precision for accuracy calculation
# There are no really accuracy factor for regression problem,
# but we can simulate/approximate it. We'll calculate it by checking
# how many values have a difference to their ground truth equivalent
# less than given precision
# We'll calculate this precision as a fraction of standard deviation
# of al the ground truth values
accuracy_precision = np.std(y) / 250

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

    # Perform a forward pass through third Dense Layer
    # Takes outputs of activation function of second layer as inputs
    dense3.forward(activation2.output)

    # Perform forward pass through activation function
    # takes the output of third layer here
    activation3.forward(dense3.output)

    # calculate data loss 
    data_loss = loss_function.calculate(activation3.output, y)

    # Calculate regularization penalty
    regularization_loss = loss_function.regularization_loss(dense1) + \
                          loss_function.regularization_loss(dense2) + \
                          loss_function.regularization_loss(dense3)

    # Calculate overall loss
    loss = data_loss + regularization_loss

    # Calculate accuracy from output of activation2 and targets
    # To calculate it we're taking absolute difference between
    # predictions and ground truth values and compare if differences
    # are lower than given precision value

    predictions = activation3.output
    accuracy = np.mean(np.absolute(predictions - y) < accuracy_precision)

    if not epoch % 100:
        print(f'epoch: {epoch}, ' +
              f'acc: {accuracy:.3f}, ' +
              f'loss: {loss:.3f}, (' +
              f'data_loss: {data_loss:.3f}, ' +
              f'reg_loss: {regularization_loss:.3f}), ' +
              f'lr: {optimizer.current_learning_rate}')

    # Backward pass
    loss_function.backward(activation3.output, y)
    activation3.backward(loss_function.dinputs)
    dense3.backward(activation3.dinputs)
    activation2.backward(dense3.dinputs)
    dense2.backward(activation2.dinputs)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)


    # Update weights and biases
    optimizer.pre_update_params()
    optimizer.update_params(dense1)
    optimizer.update_params(dense2)
    optimizer.update_params(dense3)
    optimizer.post_update_params()


# Looking at the data

import matplotlib.pyplot as plt

X_test, y_test = sine_data()

dense1.forward(X_test)
activation1.forward(dense1.output)
dense2.forward(activation1.output)
activation2.forward(dense2.output)
dense3.forward(activation2.output)
activation3.forward(dense3.output)

plt.plot(X_test, y_test)
plt.plot(X_test, activation3.output)
plt.show()

# It is important to realize that changing the initial weights values could affect
# the learning process from not learning at all, to a learning state.