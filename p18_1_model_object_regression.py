# It’s beginning to make more sense to make our model an
# object itself, especially since we will want to do things like save and load this object to use for
# future prediction tasks.

from mynnfs import * # My neural network

from mydatasets import sine_data

import numpy as np

np.random.seed(0) # Same initialization 

# Create dataset
X,y = sine_data()   # 1000 samples

model = Model()

# Add layers
model.add(Layer_Dense(1,64))
model.add(Activation_ReLU())
model.add(Layer_Dense(64,64))
model.add(Activation_ReLU())
model.add(Layer_Dense(64,1))
model.add(Activation_Linear())

# Set loss and optimizer
model.set(loss=Loss_MeanSquaredError(), optimizer=Optimizer_Adam(learning_rate=0.005, decay=1e-3),
        accuracy=Accuracy_Regression())

# Finalize the model
model.finalize()

# Train the model
model.train(X, y, epochs=10000, print_every=100)