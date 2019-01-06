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
def backprop_1d_to_1d(delta, weights, output, z_vals):


    sp = activation_prime(z_vals)
    # print 'w,d,z_vals: ', prev_weights.shape, delta.shape, sp.shape, prev_activations.shape
    delta = np.dot(weights.transpose(), delta) * sp         # backprop to calculate error (delta) at layer - 1

    delta_b = delta
    delta_w = np.dot(delta, output.transpose())

    return delta_b, delta_w, delta

@numba.njit()
def backprop_1d_to_1d_final(delta, output, z_vals):

    sp = tanh_split_complex_prime(z_vals)
    delta = delta * sp

    delta_b = delta
    delta_w = np.dot(delta, output.transpose())

    return delta_b, delta_w, delta


# fc to pool
@numba.njit()
def backprop_3d_to_1d(delta, weights, output, z_vals):
    sp = activation_prime(z_vals)

    # print 'w,d,z_vals: ', prev_weights.shape, delta.shape, sp.shape, prev_activations.shape
    delta = np.dot(weights.transpose(), delta) * sp  # backprop to calculate error (delta) at layer - 1

    delta_b = delta
    depth, dim1, dim2 = output.shape
    output = output.reshape((1, depth * dim1 * dim2))
    delta_w = np.dot(delta, output)
    delta_w = delta_w.reshape((delta.shape[0], depth, dim1, dim2))

    return delta_b, delta_w, delta


# test(
@numba.njit()
def backprop_pool_to_conv(delta, weights_shape, stride, output, prev_z_vals):
    '''weights passed in are the ones between pooling and fc layer'''

    num_filters, depth, filter_size, filter_size = weights_shape

    delta_b = np.zeros((num_filters, 1)) + 0j
    delta_w = np.zeros((weights_shape)) + 0j  # you need to change the dims of weights

    # print delta_w.shape, delta_b.shape, delta.shape
    total_deltas_per_layer = (delta.shape[1]) * (delta.shape[2])
    # print 'total_deltas_per_layer', total_deltas_per_layer
    delta = delta.reshape((delta.shape[0], delta.shape[1] * delta.shape[2]))
    backprop_pool_to_conv_loop(num_filters, total_deltas_per_layer, output, filter_size, delta, delta_w, delta_b,
                               stride)

    return delta_b, delta_w


# @numba.njit()
def backprop_conv_to_pool(delta, weights, input_from_conv, max_indices, poolsize, pool_output, from_conv=False):

    x, y, z = pool_output.shape
    a, b, c, d = weights.shape

    # same for the max index matrix
    max_indices = max_indices.reshape((x, y * z, 2))

    # backprop delta from fc to pool layer

    if not from_conv:
        weights = weights.reshape((a, b * c * d))
        pool_output = pool_output.reshape((x * y * z, 1))

        # sp = pool_output #versi awb sotoy, pooling gak pake activation
        sp = activation_prime(pool_output)  # versi old
        # log.debug("## sp.shape : %s", sp.shape)
        # log.debug("## weights.transpose().shape : %s", weights.transpose().shape)
        delta = np.dot(weights.transpose(), delta) * sp  # backprop to calc delta on pooling layer
        # log.debug("## delta.shape (after) : %s", delta.shape)
        delta = delta.reshape((x, y * z))
    else:
        stride = 1
        filter_size = c
        padding = 0

        delta_temp = delta
        depth, dim1, dim2 = delta_temp.shape

        h = ((dim1 - 1) * stride) + filter_size - (2 * padding)
        w = ((dim2 - 1) * stride) + filter_size - (2 * padding)

        delta = np.zeros((x, y * z)) + 0j

        num_filters = x
        act_length1d = y * z

        ##PARALLEL
        # import time
        # start = time.time()
        backprop_conv_to_pool_loop(depth, filter_size, dim1, dim2, delta_temp, num_filters, weights, act_length1d,
                                   pool_output, delta, stride)
        # end = time.time()
        # time = end - start
        # print "TIME backprop_conv_to_pool_loop: ",time
        ##endfunction

    pool_output = pool_output.reshape((x, y * z))

    depth, height, width = input_from_conv.shape
    delta_new = np.zeros((depth, height, width)) + 0j # calc the delta on the conv layer

    backprop_conv_to_pool_loop1(depth, max_indices, input_from_conv, poolsize, pool_output, delta, width, delta_new)
    return delta_new


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

    # PARALLEL
    # import time
    # start = time.time()
    backprop_to_conv_loop(num_filters, total_deltas_per_layer, output, filter_size, delta, delta_w, delta_b, stride)
    # end = time.time()
    # time = end - start
    # print "TIME backprop_to_conv_loop: ",time

    return delta_b, delta_w






