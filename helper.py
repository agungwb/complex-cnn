import sys

import numpy as np
import numba
import math

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

@numba.njit()
def activate(z, activation):
    if activation == 1:
        return sigmoid(z)
    elif activation == 2:
        return tanh(z)
    elif activation == 3:
        return relu(z)
    else:
        return sigmoid(z)

# @numba.jit('f8(f8)', nopython=True, parallel=True)
@numba.njit()
def activate_prime(z, activation):
    if activation == 1:
        return sigmoid_prime(z)
    elif activation == 2:
        return tanh_prime(z)
    elif activation == 3:
        return relu_prime(z)
    else:
        return sigmoid_prime(z)

def csigmoid(z):
    np.exp(1j * sigmoid(np.angle(z)))

def csigmoid_prime(z):
    1j * sigmoid_prime(z) * csigmoid(z)
    return

# @numba.jit('f8(f8)', nopython=True, parallel=True)
@numba.njit()
def sigmoid(z):
    return 1.0/(1.0 + np.exp(-z))

# @numba.jit('f8(f8)', nopython=True, parallel=True)
@numba.njit()
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

@numba.njit()
def loss(desired,final):
    return 0.5*np.sum(desired-final)**2

@numba.njit()
def loss_prime(desired, final):
    return final-desired
    # return desired-final

def loss_complex(desired,final):
    return 0.5*np.sum(desired.real-final.real)**2

@numba.njit()
def tanh(z):
    a = np.exp(z)
    b = np.exp(-z)
    return (a-b)/(a+b)

# @numba.jit("f8[:](f8[:])", parallel=True)
@numba.njit()
def tanh_prime(z):
    return 1 - (tanh(z)**2)

def tanh_split_complex(z):
    return tanh(z.real) + (1j * tanh(z.imag))

# @numba.jit("c16[:](c16[:])", parallel=True)
def tanh_split_complex_prime(z):
    return tanh_prime(z.real) + (1j * tanh_prime(z.imag))

@numba.njit()
def relu(z):
    return np.maximum(z, 0)

@numba.njit()
def relu_prime(z):
    # return np.ones(z.shape)
    # z[z>=0] = 1
    # z[z<0] = 0
    # np.maximum(z,0)
    # np.minimum(z,1)

    return (z>=0).astype(z.dtype)

    # return np.where(z>=0, 1, 0)

    # dim = len(z.shape)
    #
    # if dim == 1:
    #     return relu_prime_1d(z)
    #     # return z
    # elif dim == 2:
    #     # m = z.shape[0]
    #     # n = z.shape[1]
    #     return z
    # elif dim == 3:
    #     # m = z.shape[0]
    #     # n = z.shape[1]
    #     # o = z.shape[2]
    #     return z
    # else:
    #     return z
    # elif dim == 2:
    #     x = 10
        # for i in range(x):
        #     if z[i] > 0:
        #         z[i] = 1
        #     else:
        #         z[i] = 0
    # elif len(dim) == 2:
    #     x = 10
        # x, y = z.shape
    #     for i in range(x):
    #         for j in range(y):
    #             if z[i][j] > 0:
    #                 z[i][j] = 1
    #             else:
    #                 z[i][j] = 0
    # elif len(z.shape) == 3:
    #     x, y, z = z.shape
    #     for i in range(x):
    #         for j in range(y):
    #             for k in range(z):
    #                 if z[i][j][k] > 0:
    #                     z[i][j][k] = 1
    #                 else:
    #                     z[i][j][k] = 0

    # return z

@numba.njit
def initiate_weights_conv(num_filters, depth, filter_size):
    # weights = np.random.randn(num_filters, depth, filter_size, filter_size)  # filter * depth * filter_size * filter_size
    # biases = np.random.randn(num_filters, 1)  # filter * 1

    # weights = np.random.randint(3, size=(num_filters, depth, filter_size, filter_size))
    # biases = np.random.randint(3, size=(num_filters, 1))

    weights = np.ones((num_filters, depth, filter_size, filter_size))
    biases = np.ones((num_filters, 1))

    return weights, biases

@numba.njit
def initiate_weights_fc(num_output, depth, height_in, width_in):
    # weights = np.random.randn(num_output, depth, height_in, width_in)
    # biases = np.random.randn(num_output, 1)

    # weights = np.random.randint(3, size=(num_output, depth, height_in, width_in))
    # biases = np.random.randint(3, size=(num_output, 1))

    weights = np.ones((num_output, depth, height_in, width_in)) * 2
    biases = np.ones((num_output, 1))
    return weights, biases

@numba.njit
def initiate_weights_classify(num_classes, num_inputs):
    # weights = np.random.randn(num_classes, num_inputs)
    # biases = np.random.randn(num_classes, 1)
    # weights = np.random.randint(3, size=(num_classes, num_inputs))
    # biases = np.random.randint(3, size=(num_classes, 1))
    weights = np.ones((num_classes, num_inputs))*3
    biases = np.ones((num_classes, 1))
    return weights, biases

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

@numba.njit('c16[:,:](c16[:,:])')
def rot180_complex(a):
    row, col = a.shape
    temp = np.zeros((row, col)) + 0j
    for x in range (col, 0, -1):
        for y in range (row, 0, -1):
            # print("x, y : %s, %s", x, y)
            # print("5-x, 5-y : %s, %s", col-x, row-y)
            temp[x-1][y-1]=a[row-x][col-y]
    return temp

@numba.njit('c16[:,:](i8,i8,i8,i8,i8,i8,c16[:,:])')
def delta_padded_zeros_complex(height_in, width_in, h_gap, w_gap, dim1, dim2, delta_temp):
    delta_padded_zero = np.zeros((height_in, width_in)) + 0j
    delta_padded_zero[h_gap:dim1 + h_gap, w_gap:dim2 + w_gap] = delta_temp
    return delta_padded_zero

# @numba.njit('f8[:,]')
def transpose(x):
    y = x.transpose()  # or x.T
    return y

