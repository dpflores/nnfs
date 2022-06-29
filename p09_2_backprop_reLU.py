# Backpropagation of the ReLU function using partial derivatives

x = [1.0, -2.0, 3.0] # input values
w = [-3.0, -1.0, 2.0] # weights
b = 1.0 # bias

# Multiplying nputs by weights
xw0 = x[0]*w[0]
xw1 = x[1]*w[1]
xw2 = x[2]*w[2]

# Adding weighted inputs and bias
z = xw0 + xw1 + xw2 + b

# ReLU activation function

y = max(z, 0)

print(y)

# Backward pass

# Derivative form next layer

d_value = 1

drelu_dz = d_value*(1. if z > 0 else 0.)

#print(drelu_dz)

dsum_dxw0 = 1
dsum_dxw1 = 1
dsum_dxw2 = 1

dsum_db = 1

dmul_dx0 = w[0]
dmul_dx1 = w[1]
dmul_dx2 = w[2]

dmul_dw0 = x[0]
dmul_dw1 = x[1]
dmul_dw2 = x[2]

drelu_dx0 = drelu_dz * dsum_dxw0 * dmul_dx0
drelu_dx1 = drelu_dz * dsum_dxw1 * dmul_dx1
drelu_dx2 = drelu_dz * dsum_dxw2 * dmul_dx2

drelu_dw0 = drelu_dz * dsum_dxw0 * dmul_dw0
drelu_dw1 = drelu_dz * dsum_dxw1 * dmul_dw1
drelu_dw2 = drelu_dz * dsum_dxw2 * dmul_dw2

drelu_db = drelu_dz * dsum_db

print(drelu_dx0, drelu_dw0, drelu_dx1, drelu_dw1, drelu_dx2, drelu_dw2)

dx = [drelu_dx0, drelu_dx1, drelu_dx2]      # Gradients on inputs
dw = [drelu_dw0, drelu_dw1, drelu_dw2]      # Gradients on weights
db = drelu_db                               # Gradient on bias


#Gradient descent
w[0] += -0.001 * dw[0]
w[1] += -0.001 * dw[1]
w[2] += -0.001 * dw[2]
b += -0.001 * db


# Multiplying nputs by weights
xw0 = x[0]*w[0]
xw1 = x[1]*w[1]
xw2 = x[2]*w[2]

# Adding weighted inputs and bias
z = xw0 + xw1 + xw2 + b

# ReLU activation function

y = max(z, 0)

print(y)  # It can be seen that we reduced the output, in practice we'll pretend to reduce the loss 
