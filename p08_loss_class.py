# Implementing loss

import numpy as np

from mydatasets import spiral_data

np.random.seed(0) # Para que siempre tenga la misma inicializacion


# OOP
class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

class Activation_Softmax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True)) # subtracts the max of each row
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True) # normalize
        self.output = probabilities

class Loss:
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses) # Gets the mean of all the losses for each sample
        return data_loss

class Loss_CategoricalCrossentropy(Loss): #Inherited from the Loss Class
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7) # We present this range to not deal with infinity loss

        if len(y_true.shape) == 1: # If it is 1D array
            correct_confidences = y_pred_clipped[range(samples), y_true] # Getting confidences if user sends a 1D array of y desired
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped*y_true, axis=1) # Getting for 2D array
        negative_log_likelihoods = -np.log(correct_confidences) # Gets the loss for each sample
        return negative_log_likelihoods

X, y = spiral_data(samples=100, classes=3)

dense1 = Layer_Dense(2,3)
activation1 = Activation_ReLU()

dense2 = Layer_Dense(3, 3) # output layer
activation2 = Activation_Softmax()

dense1.forward(X)
activation1.forward(dense1.output)

dense2.forward(activation1.output)
activation2.forward(dense2.output)

#print(activation2.output[:5])

loss_function = Loss_CategoricalCrossentropy()
loss = loss_function.calculate(activation2.output, y)
print("Loss: ", loss)
