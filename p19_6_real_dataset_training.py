# Lets train a model with teh real dataset
import numpy as np
import cv2
import os
from mynnfs import *

np.random.seed(0) # Same initialization 

# Loads a MNIST dataset
def load_mnist_dataset(dataset, path):

    # Scan all the directories and create a list of labels
    labels = os.listdir(os.path.join(path,dataset))

    # Create lists for samples and labels
    X = []
    y = []

    # For each label folders
    for label in labels:
        # And for each image in teh folders
        for file in os.listdir(os.path.join(path,dataset,label)):
            # Read the image
            image = cv2.imread(os.path.join(path,dataset,label, file), cv2.IMREAD_UNCHANGED)

            # And append it and the label to the lists
            X.append(image)
            y.append(label)
        
    # Convert the data to proper numpy arrays and return
    return np.array(X), np.array(y).astype('uint8')

# MNIST dataset (train + test)
def create_data_mnist(path):

    # Load both sets separately
    X, y = load_mnist_dataset('train', path)
    X_test, y_test = load_mnist_dataset('test', path)

    # And return all the data

    return X, y, X_test, y_test



# With these functions we can create the data using
X, y, X_test, y_test = create_data_mnist('fashion_mnist_images')

# SUFFLING

# When working with batches, it is important to shuffle the data to avoid 
# biased sequential training. We need to shuffle the data and labels arrays
# We can not simply shuffle both separately. There are many ways to do this,
# but what we’ll do is gather all of the “keys,” which are the same for samples 
# and targets, and then shuffle them.

# Shuffling the data with indexes, to equally shuffle both data and labels
keys = np.array(range(X.shape[0]))
np.random.shuffle(keys)
X = X[keys]
y = y[keys]


# Scale and reshape samples
X = (X.reshape(X.shape[0],-1 ).astype(np.float32)-127.5)/127.5
X_test = (X_test.reshape(X_test.shape[0], -1).astype(np.float32) - 127.5)/127.5

# 2 hidden layers using ReLU activation, an ouput layer with softmax activation.
# since it is classification model, cross-entropy loss, Adam optimizer and categorical accuracy.

# Instantiate the model
model = Model()

# Add layers
model.add(Layer_Dense(X.shape[1], 64))
model.add(Activation_ReLU())
model.add(Layer_Dense(64, 64))
model.add(Activation_ReLU())
model.add(Layer_Dense(64 , 10))
model.add(Activation_Softmax())

# Set loss, optimizer and accuracy objects
model.set(
    loss = Loss_CategoricalCrossentropy(),
    optimizer = Optimizer_Adam(decay = 5e-5),
    accuracy = Accuracy_Categorical()
)

# Finalize the model
model.finalize()


# Train the model
model.train(X, y, validation_data=(X_test, y_test), epochs=5 , batch_size=128 , print_every=100)