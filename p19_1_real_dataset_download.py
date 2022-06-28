# Lets use a real dataset introduced by nnfs.io (MNIST Fashion)

import os
import urllib
import urllib.request
from zipfile import ZipFile

URL = 'https://nnfs.io/datasets/fashion_mnist_images.zip'
FILE = 'fashion_mnist_images.zip'
FOLDER = 'fashion_mnist_images'

# Gets the dataset from the link if already not downloaded
if not os.path.isfile(FILE):
    print(f'Downloading {URL} and saving as {FILE} ...')
    urllib.request.urlretrieve(URL, FILE)

# Unzip the dataset in the folder required

print('Unzipping images...')
with ZipFile(FILE) as zip_images:
    zip_images.extractall(FOLDER)

print('Done!')