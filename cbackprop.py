from chelper import *
from cnumba_helper import *

import logging

import numpy as np
import numba

# import time

# try:
#     import cupy as np
# except ImportError:
#     import numpy as np

log = logging.getLogger("__backprop__")


# backpropagation
##############################################################

# delta_L = weights_L (dot) delta_L+1 .*  activation_prime(z_L)


@numba.njit()
def backprop_1d_to_1d(delta, weights, output, z_vals, activation):
    sp = activate_prime(z_vals, activation)

    # delta = np.dot(weights.transpose(), delta) * sp
    delta = ((np.dot(weights.real.transpose(), delta.real) + np.dot(weights.imag.transpose(), delta.imag)) * sp.real) \
            + (1j * ((np.dot(weights.real.transpose(), delta.imag) - np.dot(weights.imag.transpose(), delta.real))) * sp.imag )

    delta_w = np.dot(delta, output.transpose().conj())
    delta_b = delta

    return delta_b, delta_w, delta

@numba.njit()
def backprop_1d_to_1d_final(loss_prime, output, z_vals, activation):
    sp = activate_prime(z_vals, activation)

    # delta = loss_prime * sp

    delta = (loss_prime.real * sp.real) + (1j * loss_prime.imag * sp.imag)


    delta_w = np.dot(delta, output.transpose().conj())
    delta_b = delta



    return delta_b, delta_w, delta


@numba.njit()
def backprop_3d_to_1d(delta, weights, output, z_vals, activation):
    sp = activate_prime(z_vals, activation)

    # delta = np.dot(weights.transpose(), delta) * sp
    delta = ((np.dot(weights.real.transpose(), delta.real) + np.dot(weights.imag.transpose(), delta.imag)) * sp.real) \
            + (1j * ((np.dot(weights.real.transpose(), delta.imag) - np.dot(weights.imag.transpose(), delta.real))) * sp.imag)

    depth, dim1, dim2 = output.shape
    output = output.reshape((1, depth * dim1 * dim2))
    delta_w = np.dot(delta, output.conj())
    delta_w = delta_w.reshape((delta.shape[0], depth, dim1, dim2))

    delta_b = delta

    return delta_b, delta_w, delta

@numba.njit()
def backprop_3d_to_3d(delta, weights, output, z_vals, activation):

    sp = activate_prime(z_vals, activation)

    # delta = np.dot(weights.transpose(), delta) * sp
    delta = ((np.dot(weights.real.transpose(), delta.real) + np.dot(weights.imag.transpose(), delta.imag)) * sp.real) \
            + (1j * ((np.dot(weights.real.transpose(), delta.imag) - np.dot(weights.imag.transpose(), delta.real))) * sp.imag)


    depth, dim1, dim2 = output.shape
    output = output.reshape((1, depth * dim1 * dim2))
    delta_w = np.dot(delta, output.conj())
    delta_w = delta_w.reshape((delta.shape[0], depth,dim1,dim2))

    delta_b = delta

    return delta_b, delta_w, delta

@numba.njit()
def backprop_conv(delta, weights_shape, stride, output, z_vals, activation):
    '''weights passed in are the ones between pooling and fc layer'''

    num_filters, depth, filter_size, filter_size = weights_shape
    sp = activate_prime(z_vals, activation)

    # delta =  sp * delta
    delta = (delta.real * sp.real) + (1j * delta.imag * sp.imag)

    delta_b = np.zeros((num_filters, 1)) + 0j
    delta_w = np.zeros((weights_shape))  + 0j # you need to change the dims of weights

    x, y, z = delta.shape

    total_deltas_per_layer = y * z  # print delta_w.shape, delta_b.shape, delta.shape
    delta = delta.reshape((x, y * z))  # print 'total_deltas_per_layer', total_deltas_per_layer
    delta, delta_b, delta_w = backprop_conv_loop(num_filters, total_deltas_per_layer, output, filter_size, delta, delta_w, delta_b,stride)

    delta = delta.reshape((x, y, z))

    return delta, delta_b, delta_w


@numba.njit()
def backprop_pool(delta, weights, input_from_conv, max_indices, poolsize, pool_output):

    x,y,z = pool_output.shape
    a,b,c,d = weights.shape

    # same for the max index matrix
    max_indices = max_indices.reshape((x, y * z, 2))


    weights = weights.reshape((a, b * c * d))
    pool_output = pool_output.reshape((x * y * z, 1))

    # sp = pool_output #versi awb sotoy, pooling gak pake activation
    # sp = activation_prime(pool_output) #versi old
    # delta = np.dot(weights.transpose(), delta) * sp         # backprop to calc delta on pooling layer

    # delta = np.dot(weights.transpose(), delta)         # versi awb without sp because there's no activation function
    delta = ((np.dot(weights.real.transpose(), delta.real) + np.dot(weights.imag.transpose(), delta.imag))) \
            + (1j * ((np.dot(weights.real.transpose(), delta.imag) - np.dot(weights.imag.transpose(), delta.real))))

    delta = delta.reshape((x, y * z))

    pool_output = pool_output.reshape((x, y * z))
    #
    # depth, height, width = input_from_conv.shape
    # delta_new = np.zeros((depth, height, width)) # calc the delta on the conv layer

    delta_new = backprop_pool_loop(input_from_conv, max_indices, poolsize, pool_output, delta)

    return delta_new

@numba.njit()
def backprop_pool_from_conv(delta, weights, input_from_conv, max_indices, poolsize, pool_output, stride, filter_size, padding):

    x,y,z = pool_output.shape

    # same for the max index matrix
    max_indices = max_indices.reshape((x, y * z, 2))

    delta_new = backprop_pool_from_conv_loop(delta, weights, pool_output, stride, filter_size)

    pool_output = pool_output.reshape((x, y * z))
    #
    # depth, height, width = input_from_conv.shape
    # delta_new = np.zeros((depth, height, width)) # calc the delta on the conv layer

    delta_new_expanded = backprop_pool_loop(input_from_conv, max_indices, poolsize, pool_output, delta_new)

    return delta_new_expanded


@numba.njit()
def backprop_to_conv(delta, weights_shape, stride, output, prev_z_vals):
    '''weights passed in are the ones between pooling and fc layer'''

    delta = delta

    # print 'weight filter, delta shape', weight_filters.shape, delta.shape
    # print 'input shape', input_to_conv.shape
    num_filters, depth, filter_size, filter_size = weights_shape

    delta_b = np.zeros((num_filters, 1), dtype=np.complex128)
    delta_w = np.zeros((weights_shape), dtype=np.complex128)  # you need to change the dims of weights

    # print delta_w.shape, delta_b.shape, delta.shape
    total_deltas_per_layer = (delta.shape[1]) * (delta.shape[2])
    # print 'total_deltas_per_layer', total_deltas_per_layer
    delta = delta.reshape((delta.shape[0], delta.shape[1] * delta.shape[2]))

    backprop_to_conv_loop(num_filters, total_deltas_per_layer, output, filter_size, delta, delta_w, delta_b, stride)

    return delta_b, delta_w






