# Calculatin loss with categorical cross-entropy

import math

softmax_output = [0.7, 0.1, 0.2] # Outputs from the output layer
target_output = [1, 0, 0] # the desired output

loss = -(math.log(softmax_output[0])*target_output[0] +  #loss using cross entropy
         math.log(softmax_output[1])*target_output[1] +
         math.log(softmax_output[2])*target_output[2])

print(loss)

