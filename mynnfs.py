#from matplotlib.font_manager import _Weight
from doctest import OutputChecker
from random import sample
import numpy as np


# OOP
class Layer_Dense:
    def __init__(self, n_inputs, n_neurons, weight_regularizer_l1=0, weight_regularizer_l2=0, 
                bias_regularizer_l1=0, bias_regularizer_l2=0):
        # Initialize weights and biases 
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
        # Set regularization strength
        self.weight_regularizer_l1 = weight_regularizer_l1
        self.weight_regularizer_l2 = weight_regularizer_l2
        self.bias_regularizer_l1 = bias_regularizer_l1
        self.bias_regularizer_l2 = bias_regularizer_l2

    def forward(self, inputs):
        # Remember input values
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases
    def backward(self, dvalues):
        # Gradients on parameters
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)

        # Gradients on regularization
        # For L1, lambda1 * abs'(x) = {1 x>0; -1 x<0}
        # For L2, lammda2 * sum(x^2)' = 2 * x
        # L1 on weights  
        if self.weight_regularizer_l1 > 0:
            dL1 = np.ones_like(self.weights)
            dL1[self.weights < 0] = -1
            self.dweights += self.weight_regularizer_l1 * dL1

        # L2 on weights
        if self.weight_regularizer_l2 > 0:
            self.dweights += 2 * self.weight_regularizer_l2 * self.weights

        # L1 on biases
        if self.bias_regularizer_l1 > 0:
            dL1 = np.ones_like(self.biases)
            dL1[self.biases < 0] = -1
            self.dbiases += 2 * self.bias_regularizer_l1 * self.weights
        
        # L2 on biases
        if self.bias_regularizer_l2 > 0:
            self.dbiases += 2 * self.bias_regularizer_l2 * self.biases

        # Gradients on values
        self.dinputs = np.dot(dvalues, self.weights.T)


# Dropout 
class Layer_Dropout:
    # Init
    def __init__(self, rate):
        # Store rate, we invert it as for example for dropout
        # of 0.1 we need success rate of 0.9
        self.rate = 1 - rate
    
    # Forward pass
    def forward(self, inputs):
        # Save input values
        self.inputs = inputs
        # Generate and save scaled mask
        self.binary_mask = np.random.binomial(1, self.rate, size=inputs.shape) / self.rate
        # Apply mask to output values
        self.output = inputs * self.binary_mask

    # Backward pass
    def backward(self, dvalues):
        # Gradient on values
        self.dinputs = dvalues * self.binary_mask


# Activation
class Activation_ReLU: # This is impotant dou to the non linearity
    # Forward pass
    def forward(self, inputs):
        # Remember input values
        self.inputs = inputs
        self.output = np.maximum(0, inputs)

    # Backward pass
    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0


class Activation_Softmax: # Important for the output layer
    # Forward pass
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True)) # subtracts the max of each row
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True) # normalize
        self.output = probabilities

    # Backward pass
    def backward(self, dvalues):
        # Create uninitialized array
        self.dinputs = np.empty_like(dvalues)
        # Enumerate outputs and gradients
        for index, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):
            # Flatten output array
            single_output = single_output.reshape(-1, 1)
            # Calculate Jacobian matrix of the output and
            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)
            # Calculate sample-wise gradient
            # and add it to the array of sample gradients
            self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)

# Sigmoid activation for binary logistic regression
class Activation_Sigmoid:
    # Forward pass
    def forward(self, inputs):
        # Sigmoid function
        self.inputs = inputs
        self.output = 1 / (1 + np.exp(-inputs))
    
    def backward(self, dvalues):
        # Derivative calculated from the output of the sigmoid function
        self.dinputs = dvalues * self.outputs * (1 - self.outputs)

class Loss:
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses) # Gets the mean of all the losses for each sample
        return data_loss

    # Regularization loss calculation
    def regularization_loss(self, layer):

        # 0 by default
        regularization_loss = 0

        # L1 regularization
        # calculate only when factor greater than 0
        if layer.weight_regularizer_l1 > 0:
            regularization_loss += layer.weight_regularizer_l1 * np.sum(np.abs(layer.weights))

        # L2 regularization - weights
        if layer.weight_regularizer_l2 > 0:
            regularization_loss += layer.weight_regularizer_l2 * np.sum(np.abs(layer.weights*layer.weights))
        
        # L1 regularization - biases
        if layer.bias_regularizer_l1 > 0:
            regularization_loss += layer.bias_regularizer_l1 * np.sum(np.abs(layer.bias))
        
        # L2 regularization - biases
        if layer.bias_regularizer_l2 > 0:
            regularization_loss += layer.bias_regularizer_l2 * np.sum(np.abs(layer.biases*layer.biases))

        return regularization_loss

class Loss_CategoricalCrossentropy(Loss): #Inherited from the Loss Class
    # Forward pass
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7) # We present this range to not deal with infinity loss

        if len(y_true.shape) == 1: # If it is 1D array
            correct_confidences = y_pred_clipped[range(samples), y_true] # Getting confidences if user sends a 1D array of y desired
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped*y_true, axis=1) # Getting for 2D array
        negative_log_likelihoods = -np.log(correct_confidences) # Gets the loss for each sample
        return negative_log_likelihoods

    # Backward pass
    def backward(self, dvalues, y_true): # dvalues correspond to the predicted values in this case

        # Number of samples
        samples = len(dvalues)
        # Number of labels in every sample
        # We will use the first sample to count them
        labels = len(dvalues[0])

        # If labels are sparse, turn them into one hot vector
        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]

        # Calculate gradient
        self.dinputs = - y_true / dvalues
        # Normalize gradient (due to a number of samples)
        self.dinputs = self.dinputs / samples

# Binary cross-entropy loss
class Loss_BinaryCrossentropy(Loss):

    # Forward pass
    def forward(self, y_pred, y_true):

        # clip datato prevent division by 0
        # clip both sides to not drag mean towards any value
        y_predd_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        # Calculate sample-wise loss
        sample_losses = -(y_true * np.log(y_predd_clipped) + (1 - y_true) * np.log(1 - y_predd_clipped))
        sample_losses = np.mean(sample_losses, axis=-1)

        # Return losses
        return sample_losses
    
    # Backward pass
    def backward(self, dvalues, y_true):

        # Number of samples
        samples = len(dvalues)
        # Number of outputs in every sample
        # We'll use the first sample to count them
        outputs = len(dvalues[0])
        clipped_dvalues = np.clip(dvalues, 1e-7, 1 - 1e-7)

        # Calculate gradient
        self.dinputs = -(y_true/clipped_dvalues - (1 - y_true) / clipped_dvalues) / outputs

        # Normalize gradiente
        self.dinputs = self.dinputs / samples

# Softmax classifier - combined Softmax activation
# and cross-entropy loss for faster backward step
class Activation_Softmax_Loss_CategoricalCrossentropy():

    # Creating activation and loss function objects
    def __init__(self):
        self.activation = Activation_Softmax()
        self.loss = Loss_CategoricalCrossentropy()

    # Forward pass
    def forward(self, inputs, y_true):
        # Output layer's activation function
        self.activation.forward(inputs)
        # Set the output
        self.output = self.activation.output
        # Calculate and return loss
        return self.loss.calculate(self.output, y_true)

    # Backward pass
    def backward(self, dvalues, y_true):  #d_values is the y_predicted
        # Number of samples (i)
        samples = len(dvalues)

        # If labels are one-hot encoded.
        # turn them into discrete values or sparse labels
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis = 1) # good way to convert the array to an index

        # We copy, so we can safely modify
        self.dinputs = dvalues.copy()
        # Calculate gradient
        self.dinputs[range(samples), y_true] -= 1  # This acts as the subtraction since the value we always subtract is 1
        # Normalize the gradient
        self.dinputs = self.dinputs / samples


class Optimizer_SGD:

    # Initialize optimizer - set settings,
    # learning rate of 1. is default for this optimizer
    def __init__(self, learning_rate=1.0, decay=0., momentum= 0.):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.momentum = momentum

    # Call once before any parameter updates
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1. / (1. + self.decay * self.iterations))

    # Update parameters
    def update_params(self, layer):

        # If we use momentum
        if self.momentum:
            
            # If layer does not contain momnetum arrays, create them filled with zeros
            if not hasattr(layer, 'weight_momentums'): # hasattr checks if that attribute exist
                layer.weight_momentums = np.zeros_like(layer.weights)
                layer.bias_momentums = np.zeros_like(layer.biases)

            # Build weight updates with momentum - take previous
            # updates multiplied by retain factor and update with current gradients

            weight_updates = self.momentum*layer.weight_momentums - self.current_learning_rate*layer.dweights 
            layer.weight_momentums = weight_updates

            #Build bias updates
            bias_updates = self.momentum*layer.bias_momentums - self.current_learning_rate*layer.dbiases
            layer.bias_momentums = bias_updates
            
        # Vanilla SGD updates (as before moment update)
        else:
            weight_updates = -self.current_learning_rate*layer.dweights
            bias_updates = -self.current_learning_rate*layer.dbiases

        # Update weights and biases using either vanilla or momentum updates
        layer.weights += weight_updates
        layer.biases += bias_updates

    # Call once after any parameter updates
    def post_update_params(self):
        self.iterations += 1

class Optimizer_Adagrad:

    # Initialize optimizer - set settings,
    # learning rate of 1. is default for this optimizer
    # Epsilon is another hyperparameter to avoid division by 0
    def __init__(self, learning_rate=1.0, decay=0., epsilon=1e-7):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon

    # Call once before any parameter updates
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1. / (1. + self.decay * self.iterations))

    # Update parameters
    def update_params(self, layer):

        # If layer does not contain cache arrays.
        # create them filled with zeros
        if not hasattr(layer, 'weight_cache'): # hasattr checks if that attribute exist
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)

        # Update cache with squared current gradients
        layer.weight_cache += layer.dweights**2
        layer.bias_cache += layer.dbiases**2

        # Vanilla SGD parameter update + normalization 
        # with squared root cache
        layer.weights += -self.current_learning_rate * layer.dweights/(np.sqrt(layer.weight_cache + self.epsilon))
        layer.biases += -self.current_learning_rate * layer.dbiases/(np.sqrt(layer.bias_cache + self.epsilon))
    

    # Call once after any parameter updates
    def post_update_params(self):
        self.iterations += 1


class Optimizer_RMSprop:

    # Initialize optimizer - set settings,
    # learning rate of 1. by default could cause instant model instability, so 0.001 is good here
    # rho is the cache memory decay rate
    def __init__(self, learning_rate=0.001, decay=0., epsilon=1e-7, rho = 0.9):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.rho = rho

    # Call once before any parameter updates
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1. / (1. + self.decay * self.iterations))

    # Update parameters
    def update_params(self, layer):

        # If layer does not contain cache arrays.
        # create them filled with zeros
        if not hasattr(layer, 'weight_cache'): # hasattr checks if that attribute exist
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)

        # Update cache with squared current gradients
        layer.weight_cache = self.rho * layer.weight_cache +  (1-self.rho)*layer.dweights**2
        layer.bias_cache = self.rho * layer.bias_cache + (1-self.rho)*layer.dbiases**2

        # Vanilla SGD parameter update + normalization 
        # with squared root cache
        layer.weights += -self.current_learning_rate * layer.dweights/(np.sqrt(layer.weight_cache + self.epsilon))
        layer.biases += -self.current_learning_rate * layer.dbiases/(np.sqrt(layer.bias_cache + self.epsilon))
    

    # Call once after any parameter updates
    def post_update_params(self):
        self.iterations += 1


class Optimizer_Adam:

    # Initialize optimizer - set settings,
    # learning rate of 1. by default could cause instant model instability, so 0.001 is good here
    # rho is the cache memory decay rate

    # Beta1 and beta2 must be less than 1
    def __init__(self, learning_rate=0.001, decay=0., epsilon=1e-7, rho = 0.9, beta_1=0.9, beta_2=0.999):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.beta_1 = beta_1
        self.beta_2 = beta_2

    # Call once before any parameter updates
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1. / (1. + self.decay * self.iterations))

    # Update parameters
    def update_params(self, layer):

        # If layer does not contain cache arrays.
        # create them filled with zeros
        if not hasattr(layer, 'weight_cache'): # hasattr checks if that attribute exist
            layer.weight_momentums = np.zeros_like(layer.weights)
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_momentums = np.zeros_like(layer.biases)
            layer.bias_cache = np.zeros_like(layer.biases)
            

        # Update momentum with current gradients
        layer.weight_momentums = self.beta_1 * layer.weight_momentums + (1 - self.beta_1) * layer.dweights
        layer.bias_momentums = self.beta_1 * layer.bias_momentums + (1 - self.beta_1) * layer.dbiases

        # Get corrected momentum 
        # self.iteration is 0 at first pass
        # and we need to start with 1 here
        weight_momentums_corrected = layer.weight_momentums/(1 - self.beta_1**(self.iterations + 1))
        bias_momentums_corrected = layer.bias_momentums/(1 - self.beta_1**(self.iterations + 1))

        # Update cache with current gradients

        layer.weight_cache = self.beta_2 * layer.weight_cache + (1 - self.beta_2) * layer.dweights**2
        layer.bias_cache = self.beta_2 * layer.bias_cache + (1 - self.beta_2) * layer.dbiases**2

        # Get corrected cache 
        weight_cache_corrected = layer.weight_cache/(1 - self.beta_2**(self.iterations + 1))
        bias_cache_corrected = layer.bias_cache/(1 - self.beta_2**(self.iterations + 1))

        # Vanilla SGD parameter + normalization with squared rooted cache
        layer.weights += -self.current_learning_rate * weight_momentums_corrected/(np.sqrt(weight_cache_corrected) + self.epsilon)
        layer.biases += -self.current_learning_rate * bias_momentums_corrected/(np.sqrt(bias_cache_corrected) + self.epsilon)


    # Call once after any parameter updates
    def post_update_params(self):
        self.iterations += 1
