# Backpropagation of softmax activation function combined with categorical cross entropy loss
from timeit import timeit
from mynnfs import * # My neural network

import numpy as np

np.random.seed(0) # Para que siempre tenga la misma inicializacion

softmax_outputs = np.array([[0.7, 0.1, 0.2],
                            [0.1, 0.5, 0.4],
                            [0.02, 0.9, 0.08]])

class_targets = np.array([0, 1, 1]) # Sparse labels

# Using the combined class
def f1():
        softmax_loss = Activation_Softmax_Loss_CategoricalCrossentropy()
        softmax_loss.backward(softmax_outputs, class_targets)
        #data_loss1 = softmax_loss.forward(softmax_outputs, class_targets)
        dvalues1 = softmax_loss.dinputs

# Using them separately
def f2():
        activation = Activation_Softmax()
        activation.output = softmax_outputs
        loss = Loss_CategoricalCrossentropy()
        #data_loss2 = loss.calculate(activation.output, class_targets)
        loss.backward(activation.output, class_targets)
        activation.backward(loss.dinputs)
        dvalues2 = activation.dinputs

t1 = timeit(lambda: f1(), number=10000)
t2 = timeit(lambda: f2(), number=10000)
print(t2/t1)
