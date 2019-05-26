import sys

import numpy as np
import numba

# try:
#     import cupy as np
# except ImportError:
#     import numpy as np

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
    if sys.argv[1] == 'ccnn':
        # return sigmoid_split_complex(z)
        return tanh_split_complex(z)
    else:
        return sigmoid(z)

# @numba.jit('f8(f8)', nopython=True, parallel=True)
def activation_prime(z):
    # if sys.argv[1] == 'ccnn':
    #     # return sigmoid_split_complex_prime(z)
    #     return tanh_split_complex_prime(z)
    # else:
    return sigmoid_prime(z)

def csigmoid(z):
    np.exp(1j * sigmoid(np.angle(z)))

def csigmoid_prime(z):
    1j * sigmoid_prime(z) * csigmoid(z)
    return

# @numba.jit('f8(f8)', nopython=True, parallel=True)
def sigmoid(z):
    return 1.0/(1.0 + np.exp(-z))

# @numba.jit('f8(f8)', nopython=True, parallel=True)
def sigmoid_prime(z):
    sig = sigmoid(z)
    return sig * (1-sig)

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

def loss_prime(desired, final):
    return desired-final

def loss_complex(desired,final):
    return 0.5*np.sum(desired.real-final.real)**2

def tanh(z):
    a = np.exp(z)
    b = np.exp(-z)
    return (a-b)/(a+b)

# @numba.jit("f8[:](f8[:])", parallel=True)
def tanh_prime(z):
    return 1 - (tanh(z)**2)

def tanh_split_complex(z):
    return tanh(z.real) + (1j * tanh(z.imag))

# @numba.jit("c16[:](c16[:])", parallel=True)
def tanh_split_complex_prime(z):
    return tanh_prime(z.real) + (1j * tanh_prime(z.imag))

def relu(z):
    # print "z : ",z
    # return z if z >= 0 else 0
    return np.maximum(z, 0)

def relu_prime(z):
    # print "z : ", z
    return np.where(z>=0, 1, 0)

@numba.njit('f8[:,:](f8[:,:])')
def rot180(a):
    row, col = a.shape
    temp = np.zeros((row, col), dtype=np.float64)
    for x in range (col, 0, -1):
        for y in range (row, 0, -1):
            # print("x, y : %s, %s", x, y)
            # print("5-x, 5-y : %s, %s", col-x, row-y)
            temp[x-1][y-1]=a[row-x][col-y]
    return temp

@numba.njit('f8[:,:](i8,i8,i8,i8,i8,i8,f8[:,:])')
def delta_padded_zeros(height_in, width_in, h_gap, w_gap, dim1, dim2, delta_temp):
    delta_padded_zero = np.zeros((height_in, width_in))
    delta_padded_zero[h_gap:dim1 + h_gap, w_gap:dim2 + w_gap] = delta_temp
    return delta_padded_zero