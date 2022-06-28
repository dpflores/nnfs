from mynnfs import * # My neural network

from mydatasets import spiral_data

import numpy as np

np.random.seed(0) # Para que siempre tenga la misma inicializacion

X = np.array([1, 2, 3, 4, 5, 6])
y = np.array([7, 8, 9, 10, 11, 12])


keys = np.array(range(X.shape[0]))
np.random.shuffle(keys)
print(keys)
print(X[keys])

print(y[keys])
