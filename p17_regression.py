import matplotlib.pyplot as plt
from mynnfs import * # My neural network

from mydatasets import sine_data

import numpy as np

np.random.seed(0) # Same initialization 

X,y = sine_data()

plt.plot(X, y)
plt.show()