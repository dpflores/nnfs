# Lets implement the dropout layer
# This is to prevent a neural network from becoming too dependent on any neuron
# It also helps with co-adoption, whicha happens when neuron depend on the output of other neurons
# Also helps with noise and other perturbations in the training data.

# Dropout forces the model to use more neurons for the same purpose, with a higher chance of learning 
# the underlying function that describes the data.


from mynnfs import * # My neural network

from mydatasets import spiral_data

import numpy as np

np.random.seed(0) # Para que siempre tenga la misma inicializacion

# Create datasets

X,y = spiral_data(samples=1000, classes = 3) # two internal features

# Create Dense layer with 2 input features and 512 output values (512 neurons)
dense1 = Layer_Dense(2, 512, weight_regularizer_l2=5e-4, bias_regularizer_l2=5e-4)

# Create ReLU activation (to be used with Dense layer):
activation1 = Activation_ReLU()

# Create dropout layer
dropout1 = Layer_Dropout(0.1)

# Create second Dense layer with 512 input features (as we take output
# of previous layer here) and 3 output values (output values)
dense2 = Layer_Dense(512, 3)

# Create Softmax classifier's combined loss and activation
loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()

# Create optimizer
# The learning rate is the fraction of the gradient that we apply
# in order to descend the loss value
optimizer = Optimizer_Adam(learning_rate=0.05, decay=5e-5)

# Each full pass through all of the training data is called an epoch
for epoch in range(10001):
    # Forward pass
    # Perform a forward pass of our training data through this layer
    dense1.forward(X)

    # Perform a forward pass through activation function
    # takes the output of first dense layer here
    activation1.forward(dense1.output)

    # Perform forward pass through Dropout layer
    dropout1.forward(activation1.output)

    # Perform a forward pass through second Dense layer
    # takes outputs of activation function of first layer as inputs
    dense2.forward(dropout1.output)

    # Perform a forward pass through second Dense layer
    # takes outputs of activation function of first layer as inputs
    # We obtain the data loss 
    data_loss = loss_activation.forward(dense2.output, y)

    # Calculate regularization penalty
    regularization_loss = loss_activation.loss.regularization_loss(dense1) + loss_activation.loss.regularization_loss(dense2)

    # Calculate overall loss
    loss = data_loss + regularization_loss

    # Calculate accuracy from output of activation2 and targets
    # calculate values along first axis
    predictions = np.argmax(loss_activation.output, axis=1)
    if len(y.shape) == 2:
        y = np.argmax(y, axis=1)
    accuracy = np.mean(predictions==y)

    if not epoch % 100:
        print(f'epoch: {epoch}, ' +
              f'acc: {accuracy:.3f}, ' +
              f'loss: {loss:.3f}, (' +
              f'data_loss: {data_loss:.3f}, ' +
              f'reg_loss: {regularization_loss:.3f}), ' +
              f'lr: {optimizer.current_learning_rate}')

    # Backward pass
    loss_activation.backward(loss_activation.output, y)
    dense2.backward(loss_activation.dinputs)
    dropout1.backward(dense2.dinputs)
    activation1.backward(dropout1.dinputs)
    dense1.backward(activation1.dinputs)


    # Update weights and biases
    optimizer.pre_update_params()
    optimizer.update_params(dense1)
    optimizer.update_params(dense2)
    optimizer.post_update_params()



# Validate the model

# Create test dataset
X_test, y_test = spiral_data(samples=100,classes=3)

# Perform a forward pass of our testing data through this layer
dense1.forward(X_test)

# Perform a forward pass through activation function
# takes the output of first dense layer here
activation1.forward(dense1.output)

# Perform a forward pass through second Dense layer
# takes outputs of activations function of first layer as inputs
dense2.forward(activation1.output)

# Perform a forward pass through the activation/loss function
# takes the output of second dense layer here and return loss
loss = loss_activation.forward(dense2.output, y_test)

# Calculate accuracy from output of activation2 and targets 
# calculate values along first axis
predictions = np.argmax(loss_activation.output, axis=1)
if len(y_test.shape) == 2:
    y_test = np.argmax(y_test, axis=1)
accuracy = np.mean(predictions==y_test)

print(f'validation, acc: {accuracy:.3f}, loss: {loss:.3f}')

# A difference of 10% approximately is a common sign of serious overfitting from sentdex experience

# Pretty good result, but worse compared to the “no dropout” model. Interestingly, validation
# accuracy is close to the training accuracy with dropout — usually validation accuracy will be
# higher, so we might suspect these as signs of overfitting here