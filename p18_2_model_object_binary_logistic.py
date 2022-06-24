# It’s beginning to make more sense to make our model an
# object itself, especially since we will want to do things like save and load this object to use for
# future prediction tasks.

from mynnfs2 import * # My neural network

from mydatasets import sine_data, spiral_data

import numpy as np

np.random.seed(0) # Same initialization 

# Create dataset
X,y = spiral_data(samples=100, classes=2)  
X_test, y_test = spiral_data(samples=100, classes=2)

# Reshape labels to be a list of lists
# Inner list contains one output (either 0 or 1)
# per each output neuron, 1 in this case
y = y.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)

# Init the model
model = Model()

# Add layers
model.add(Layer_Dense(2, 64, weight_regularizer_l2=5e-4,bias_regularizer_l2=5e-4))
model.add(Activation_ReLU())
model.add(Layer_Dense(64,1))
model.add(Activation_Sigmoid())

# Set loss and optimizer
model.set(loss=Loss_BinaryCrossentropy(), 
        optimizer=Optimizer_Adam(learning_rate=0.001,  decay=5e-5),
        accuracy=Accuracy_Categorical())

# Finalize the model
model.finalize()

# Train the model
model.train(X, y, validation_data=(X_test,y_test), epochs=10000, print_every=100)