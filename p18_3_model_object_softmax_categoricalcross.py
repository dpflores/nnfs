# It’s beginning to make more sense to make our model an
# object itself, especially since we will want to do things like save and load this object to use for
# future prediction tasks.

from mynnfs2 import * # My neural network

from mydatasets import sine_data, spiral_data

import numpy as np

np.random.seed(0) # Same initialization 

# Create dataset
X,y = spiral_data(samples=1000, classes=3)  
X_test, y_test = spiral_data(samples=100, classes=3)

# Init the model
model = Model()

# Add layers
model.add(Layer_Dense(2, 512, weight_regularizer_l2=5e-4,bias_regularizer_l2=5e-4))
model.add(Activation_ReLU())
model.add(Layer_Dropout(0.1))
model.add(Layer_Dense(512,3))
model.add(Activation_Softmax())

# Set loss and optimizer
model.set(loss=Loss_CategoricalCrossentropy(), 
        optimizer=Optimizer_Adam(learning_rate=0.05, decay=5e-5),
        accuracy=Accuracy_Categorical())

# Finalize the model
model.finalize()

# Train the model
model.train(X, y, validation_data=(X_test, y_test), epochs=10000, print_every=100)


# Now that we’ve got this Model class, we’re
# able to define new models without writing large amounts of code repeatedly. Rewriting code is
# annoying and leaves more room to make small, hard-to-notice mistakes.