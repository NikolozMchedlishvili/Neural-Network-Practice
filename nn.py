X = [1, 2.1, 3.7] # Input layer (3 neurons)

W1 = [ # 1 Hidden layer (3 inputs to 4 neurons)
    [0.2, 2.4, -3.6],
    [-0.9, 1.0, 0.5],
    [2.1, -1.3, 0.3],
    [0.7, 1.5, -0.8]
]

W2 = [ # Output layer (4 inputs to 2 neurons)
    [-1.7, 1.0, -2.8, 3.0],
    [1.4, -0.5, 2.3, -1.0]
]

B1 = [2, 5, 6, 8] # Biases for hidden layer (4 neurons)
B2 = [1, 3] # Biases for output layer (2 neurons)



def dot_product(A, B):
    result = []
    for A_row in A: # A_row = W1, matrix
        dot_product = 0
        for a, b in zip(A_row, B): # Zip matches X to Weight's corresponding element (0.2 to 1)
            dot_product += a * b # a = element in weight, b = element in X
        result.append(dot_product)
    return result

def add_bias(A, B): # Z1 = list of dot products or vector of floats
    result = []
    for a, b in zip(A, B):
        result.append(a + b)
    return result

def relu(x): # Activation function
    result = []
    for i in x:
        if i > 0:
            result.append(i)
        else:
            result.append(0)
    return result

# Z1 = dot product + bias
Z1 = dot_product(W1, X)
Z1 = add_bias(Z1, B1)
A1 = relu(Z1) # A1 = Z1 in Activation function / output of hidden layer

Z2 = dot_product(W2, X)
Z2 = add_bias(Z2, B2)
A2 = relu(Z2) # Output layer

print(A2) # Print Output layer