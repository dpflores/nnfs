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



