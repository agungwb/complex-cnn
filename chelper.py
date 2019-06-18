import sys

import numpy as np
import numba

# try:
#     import cupy as np
# except ImportError:
#     import numpy as np


############## ACTIVATION ##############

@numba.njit()
def activate(z, activation):
    if activation == 1:
        return sigmoid_split(z)
    elif activation == 2:
        return tanh_split(z)
    elif activation == 3:
        return relu_split(z)
    else:
        return sigmoid_split(z)

@numba.njit()
def activate_prime(z, activation):
    if activation == 1:
        return sigmoid_split_prime(z)
    elif activation == 2:
        return tanh_split_prime(z)
    elif activation == 3:
        return relu_split_prime(z)
    else:
        return sigmoid_split_prime(z)

@numba.njit()
def sigmoid(z):
    return 1.0/(1.0 + np.exp(-z))

@numba.njit()
def sigmoid_prime(z):
    sig = sigmoid(z)
    return sig * (1-sig)

@numba.njit()
def sigmoid_split(z):
    return sigmoid(z.real) + (1j * sigmoid(z.imag))

@numba.njit()
def sigmoid_split_prime(z):
    return sigmoid_prime(z.real) + (1j * sigmoid_prime(z.imag))

@numba.njit()
def sigmoid_split_complex(z):
    return sigmoid(z.real) + (1j * sigmoid(z.imag))

@numba.njit()
def sigmoid_angle(z):
    np.exp(1j * sigmoid(np.angle(z)))

@numba.njit()
def sigmoid_angle_prime(z):
    1j * sigmoid_prime(z) * sigmoid_angle(z)
    return

@numba.njit()
def tanh(z):
    a = np.exp(z)
    b = np.exp(-z)
    return (a-b)/(a+b)

@numba.njit()
def tanh_prime(z):
    return 1 - (tanh(z)**2)

@numba.njit()
def tanh_split(z):
    return tanh(z.real) + (1j * tanh(z.imag))

@numba.njit()
def tanh_split_prime(z):
    return tanh_prime(z.real) + (1j * tanh_prime(z.imag))

@numba.njit()
def relu(z):
    return np.maximum(z, 0)

@numba.njit()
def relu_prime(z):
    return (z >= 0).astype(z.dtype)

@numba.njit()
def relu_split(z):
    return relu(z.real) + 1j * relu(z.imag)

@numba.njit()
def relu_split_prime(z):
    return relu_prime(z.real) + 1j * relu_prime(z.imag)

# @numba.njit()
def loss(predicted, final, loss_function):
    if loss_function == 1:
        return quadratic_loss(predicted, final)
    elif loss_function == 2:
        return binary_cross_entropy_loss(predicted, final)

# @numba.njit()
def loss_prime(predicted, final, loss_function):
    if loss_function == 1:
        return quadratic_loss_prime(predicted, final)
    elif loss_function == 2:
        return binary_cross_entropy_loss_prime(predicted, final)

@numba.njit()
def quadratic_loss(predicted,final):
    return 0.5 * (np.sum(predicted.real-final)**2 + np.sum(predicted.imag)**2)

@numba.njit()
def quadratic_loss_prime(predicted, final):
    return predicted-final
    # return (predicted.real-final) + (1j *predicted.imag)

# @numba.njit()
def binary_cross_entropy_loss(predicted, final):
    if final[0] == 1+1j:
        return -(np.log(predicted.real) + (1j * np.log(predicted.imag)))
    else:
        return -(np.log(1 - predicted.real) + (1j * np.log(1 - predicted.imag)))

# @numba.njit()
def binary_cross_entropy_loss_prime(predicted, final):
    if final[0] == 1+1j:
        return -((float(1)/predicted.real) + (1j * (float(1)/predicted.imag)))
    else:
        return (float(1)/(1-predicted).real) + (1j * float(1)/(1-predicted.imag))

def loss_complex_prime(desired, final):
    return desired.real - final.real



@numba.njit('c16[:,:](c16[:,:])')
def rot180(a):
    row, col = a.shape
    temp = np.zeros((row, col)) + 0j
    for x in range (col, 0, -1):
        for y in range (row, 0, -1):
            # print("x, y : %s, %s", x, y)
            # print("5-x, 5-y : %s, %s", col-x, row-y)
            temp[x-1][y-1]=a[row-x][col-y]
    return temp

@numba.njit('c16[:,:](i8,i8,i8,i8,i8,i8,c16[:,:])')
def delta_padded_zeros(height_in, width_in, h_gap, w_gap, dim1, dim2, delta_temp):
    delta_padded_zero = np.zeros((height_in, width_in)) + 0j
    delta_padded_zero[h_gap:dim1 + h_gap, w_gap:dim2 + w_gap] = delta_temp
    return delta_padded_zero

@numba.njit()
def transpose(x):
    y = x.transpose()  # or x.T
    return y


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

@numba.njit
def initiate_weights_conv(num_filters, depth, filter_size):
    # self.weights = np.random.randn(self.num_filters, self.depth, self.filter_size,self.filter_size) + 1j * np.random.randn(self.num_filters, self.depth,
    #                                                                         self.filter_size,
    #                                                                         self.filter_size)  # filter * depth * filter_size * filter_size
    # self.biases = np.random.rand(self.num_filters, 1) + 1j * np.random.rand(self.num_filters, 1)  # filter * 1

    weights = (np.random.randn(num_filters, depth, filter_size, filter_size * 2).view(np.complex128) * np.sqrt(2)) / np.sqrt(depth) # filter * depth * filter_size * filter_size
    biases = np.random.rand(num_filters, 1 * 2).view(np.complex128)  # filter * 1

    # weights = np.random.randn(num_filters, depth, filter_size, filter_size) * np.sqrt(2 / (depth)) + \
    #           (1j * np.random.randn(num_filters, depth, filter_size, filter_size) * np.sqrt(2 / (depth)))
    # biases = np.random.randn(num_filters, 1)  + \
    #          (1j * np.random.randn(num_filters, 1))

    # weights = np.random.randn(num_filters, depth, filter_size, filter_size * 2).view(np.complex128) * np.sqrt(2 / (depth))  # filter * depth * filter_size * filter_size
    # biases = np.random.rand(num_filters, 1 * 2).view(np.complex128)  # filter * 1

    # weights = np.random.randint(3, size=(num_filters, depth, filter_size, filter_size))
    # biases = np.random.randint(3, size=(num_filters, 1))

    # weights = np.ones((num_filters, depth, filter_size, filter_size)) + 1j
    # biases = np.ones((num_filters, 1)) + 1j

    return weights, biases

@numba.njit
def initiate_weights_fc(num_output, depth, height_in, width_in):
    # self.weights = np.random.randn(self.num_output, self.depth, self.height_in, self.width_in) + 1j * np.random.randn(self.num_output, self.depth, self.height_in, self.width_in)
    # self.biases = np.random.randn(self.num_output, 1) + 1j * np.random.randn(self.num_output, 1)

    weights = (np.random.randn(num_output, depth, height_in, width_in * 2).view(np.complex128) * np.sqrt(2)) / np.sqrt(depth)
    biases = np.random.randn(num_output, 1 * 2).view(np.complex128)

    # weights = np.random.randn(num_output, depth, height_in, width_in) * np.sqrt(2 / (depth)) + \
    #           (1j * np.random.randn(num_output, depth, height_in, width_in) * np.sqrt(2 / (depth)))
    # biases = np.random.randn(num_output, 1) + \
    #          (1j * np.random.randn(num_output, 1))

    # weights = np.random.randn(num_output, depth, height_in, width_in * 2).view(np.complex128) * np.sqrt(2 / (depth))
    # biases = np.random.randn(num_output, 1 * 2).view(np.complex128)

    # weights = np.random.randint(3, size=(num_output, depth, height_in, width_in))
    # biases = np.random.randint(3, size=(num_output, 1))

    # weights = np.ones((num_output, depth, height_in, width_in)) + 1j
    # biases = np.ones((num_output, 1)) + 1j
    return weights, biases

@numba.njit
def initiate_weights_classify(num_classes, num_inputs):
    # self.weights = np.random.randn(self.num_classes, num_inputs) + 1j * np.random.randn(self.num_classes, num_inputs)
    # self.biases = np.random.randn(self.num_classes, 1) + 1j * np.random.randn(self.num_classes, 1)

    weights = (np.random.randn(num_classes, num_inputs * 2).view(np.complex128) * np.sqrt(2)) / np.sqrt(num_inputs)
    biases = np.random.randn(num_classes, 1 * 2).view(np.complex128)

    # weights = np.random.randn(num_classes, num_inputs) * np.sqrt(2 / num_inputs) + \
    #           (1j * np.random.randn(num_classes, num_inputs) * np.sqrt(2 / num_inputs))
    # biases = np.random.randn(num_classes, 1) + \
    #          (1j * np.random.randn(num_classes, 1))

    # weights = np.random.randn(num_classes, num_inputs * 2).view(np.complex128) * np.sqrt(2 / (num_inputs))
    # biases = np.random.randn(num_classes, 1 * 2).view(np.complex128)

    # weights = np.random.randint(3, size=(num_classes, num_inputs))
    # biases = np.random.randint(3, size=(num_classes, 1))

    # weights = np.ones((num_classes, num_inputs)) + 1j
    # biases = np.ones((num_classes, 1)) + 1j
    return weights, biases



