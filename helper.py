import numpy as np

# helper functions
###############################################################
def max_prime(res, delta, tile_to_pool):
    dim1, dim2 = tile_to_pool.shape
    tile_to_pool = tile_to_pool.reshape((dim1 * dim2))
    new_delta = np.zeros((tile_to_pool.shape))
    for i in range(len(tile_to_pool)):
        num = tile_to_pool[i]
        if num < res:
            new_delta[i] = 0
        else:
            new_delta[i] = delta
    return new_delta.reshape((dim1, dim2))

def cross_entropy(batch_size, output, expected_output):
    return (-1/batch_size) * np.sum(expected_output * np.log(output) + (1 - expected_output) * np.log(1-output))

def activation(z):
    # return relu(z)
    # return tanh(z)
    # return sigmoid(z)
    return sigmoid_complex(z)
    # return csigmoid(z)

def activation_prime(z):
    # return relu_prime(z)
    # return tanh_prime(z)
    # return sigmoid_prime(z)
    return sigmoid_complex_prime(z)
    # return csigmoid_prime(z)

def csigmoid(z):
    np.exp(1j * sigmoid(np.angle(z)))

def csigmoid_prime(z):
    1j * sigmoid_prime(z) * csigmoid(z)
    return

def sigmoid(z):
    return 1.0/(1.0 + np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z) * (1-sigmoid(z))

def sigmoid_complex(z):
    return sigmoid(z.real) + (1j * z.imag)

def sigmoid_complex_prime(z):
    return sigmoid_prime(z.real) + (1j * z.imag)

def sigmoid_split_complex(z):
    return sigmoid(z.real) + (1j * sigmoid(z.imag))

def sigmoid_split_complex_prime(z):
    return sigmoid_prime(z.real) + (1j * sigmoid_prime(z.imag))

def loss(desired,final):
    return 0.5*np.sum(desired-final)**2

def tanh(z):
    a = np.exp(z)
    b = np.exp(-z)
    return (a-b)/(a+b)

def tanh_prime(z):
    return 1 - (tanh(z)**2)

def relu(z):
    # print "z : ",z
    # return z if z >= 0 else 0
    return np.maximum(z, 0)

def relu_prime(z):
    print "z : ", z
    return np.where(z>=0, 1, 0)