# With the data downloaded (p19_1 already executed), let's read the images and watch them
import os

labels = os.listdir('fashion_mnist_images/train')

print(labels)

files = os.listdir( 'fashion_mnist_images/train/0' )
print (files[: 10 ])
print ( len (files))
# We have 6000 samples for each class

# Lets watch an image in 2D array
import numpy as np 
# To watch the printed smaller
np.set_printoptions( linewidth = 200 )
import cv2
# cv2.IMREAD_UNCHANGED to keep the grayscale
image_data = cv2.imread( 'fashion_mnist_images/train/7/0002.png', cv2.IMREAD_UNCHANGED)
print (image_data)

# Now watch another image using matplotlib with grayscale
import matplotlib.pyplot as plt
image_data = cv2.imread( 'fashion_mnist_images/train/4/0011.png' ,cv2.IMREAD_UNCHANGED)
plt.imshow(image_data, cmap='gray')
plt.show()