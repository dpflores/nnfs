# Backpropagation of softmax activation function

softmax_output = [0.7, 0.1, 0.2]

import numpy as np

softmax_output = np.array(softmax_output).reshape(-1, 1) # Important working as matrix
#print(softmax_output)


# We have two sides of the derivative

# First in the left, multiplied by the Kronecker delta (As we have that the Kronecker delta will be 1 when j = k, we'll have a diagonal matrix)
#print(softmax_output * np.eye(softmax_output.shape[0]))
# A simplification of this is the method
#print(np.diagflat(softmax_output))

# The other part, we just have a multiplication that iterates over j and k, so we can use the dot product by transpose to get that
#print(np.dot(softmax_output,softmax_output.T))

# Finally, we can perform the subtraction of both 2D arrays (as our equation)

#print(np.diagflat(softmax_output) - np.dot(softmax_output,softmax_output.T))

#IMPLEMENTATION

softmax_output = np.array([[0.4, 0.1, 0.2],
                           [0.7, 0.1, 0.2]])
dvalues = np.array([[1., 0.5, 1.],
                    [2., 2., 2.]])  # 2D



dinputs = np.empty_like(dvalues)
for index, (single_output, single_dvalues) in enumerate(zip(softmax_output, dvalues)):
        # Flatten output array
        single_output = single_output.reshape(-1, 1)
        # Calculate Jacobian matrix of the output and
        jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)
        # Calculate sample-wise gradient
        # and add it to the array of sample gradients
        print(jacobian_matrix)
        print(single_dvalues)
        dinputs[index] = np.dot(jacobian_matrix, single_dvalues)

print(dinputs)
