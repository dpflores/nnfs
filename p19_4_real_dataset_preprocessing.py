# Now we need to porperly load and create the data for training and testing 
import numpy as np
import cv2
import os

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


# PREPROCESSING

# We will scale the data (not the image) since neural networks
# tend to work best with data in the range of either 0 to 1 or -1 to 1, 
# here the image data are in the range of 0 to 255 
# here we will convert to range -1 to 1

# Scale features (the image data was in int8 for 0 to 255, 
# so we convert to float32)
X = (X.astype(np.float32) - 127.5) / 127.5
X_test = (X_test.astype(np.float32) - 127.5) / 127.5

# Ensure that you scale both training and testing data using identical methods. Later, when making
# predictions, you will also need to scale the input data for inference.

# Any preprocessing rules should be derived without knowledge of the 
# testing dataset, but then applied to the testing set
# For example, your entire dataset might have a min value of 0 and
# a max of 125, while the training dataset only has a min of 0 and a max of 100.
# You will still use the 100 value when scaling your testing dataset.

print (X.min(), X.max())
print(X.shape)

# As the neural network dont accept 2D data (28x28) like convolutional neural network, we need to
# faltten them, but keeping the 60000 samples

# Reshape to vectors
X = X.reshape(X.shape[ 0 ], - 1 )
X_test = X_test.reshape(X_test.shape[ 0 ], - 1 )

