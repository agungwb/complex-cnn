import sys

import numpy as np
import numba

# try:
#     import cupy as np
# except ImportError:
#     import numpy as np

# helper functions
###############################################################
@jit(nopython=True)
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

@jit(nopython=True)
def cross_entropy(batch_size, output, expected_output):
    return (-1/batch_size) * np.sum(expected_output * np.log(output) + (1 - expected_output) * np.log(1-output))

@jit(nopython=True)
def activation(z):
    if sys.argv[1] == 'ccnn':
        # return sigmoid_split_complex(z)
        return tanh_split_complex(z)
    else:
        return sigmoid(z)

@jit(nopython=True)
def activation_prime(z):
    if sys.argv[1] == 'ccnn':
        # return sigmoid_split_complex_prime(z)
        return tanh_split_complex_prime(z)
    else:
        return sigmoid_prime(z)

@jit(nopython=True)
def csigmoid(z):
    np.exp(1j * sigmoid(np.angle(z)))

@jit(nopython=True)
def csigmoid_prime(z):
    1j * sigmoid_prime(z) * csigmoid(z)
    return

@jit(nopython=True)
def sigmoid(z):
    return 1.0/(1.0 + np.exp(-z))

@jit(nopython=True)
def sigmoid_prime(z):
    return sigmoid(z) * (1-sigmoid(z))

@jit(nopython=True)
def sigmoid_complex(z):
    return sigmoid(z.real) + (1j * z.imag)

@jit(nopython=True)
def sigmoid_complex_prime(z):
    return sigmoid_prime(z.real) + (1j * z.imag)

@jit(nopython=True)
def sigmoid_split_complex(z):
    return sigmoid(z.real) + (1j * sigmoid(z.imag))

@jit(nopython=True)
def sigmoid_split_complex_prime(z):
    return sigmoid_prime(z.real) + (1j * sigmoid_prime(z.imag))

@jit(nopython=True)
def loss(desired,final):
    return 0.5*np.sum(desired-final)**2

@jit(nopython=True)
def loss_prime(desired, final):
    return desired-final

@jit(nopython=True)
def loss_complex(desired,final):
    return 0.5*np.sum(desired.real-final.real)**2

@jit(nopython=True)
def tanh(z):
    a = np.exp(z)
    b = np.exp(-z)
    return (a-b)/(a+b)

@jit(nopython=True)
def tanh_prime(z):
    return 1 - (tanh(z)**2)

@jit(nopython=True)
def tanh_split_complex(z):
    return tanh(z.real) + (1j * tanh(z.imag))

@jit(nopython=True)
def tanh_split_complex_prime(z):
    return tanh_prime(z.real) + (1j * tanh_prime(z.imag))

@jit(nopython=True)
def relu(z):
    # print "z : ",z
    # return z if z >= 0 else 0
    return np.maximum(z, 0)

@jit(nopython=True)
def relu_prime(z):
    # print "z : ", z
    return np.where(z>=0, 1, 0)