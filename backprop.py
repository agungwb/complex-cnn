from helper import *
from numba_helper import *

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
    # log.debug("## delta.shape : %s", delta.shape)
    # log.debug("## weights.shape : %s", weights)
    # log.debug("## z_vals.shape : %s", z_vals.shape)

    sp = activation_prime(z_vals)
    # print 'w,d,z_vals: ', prev_weights.shape, delta.shape, sp.shape, prev_activations.shape
    delta = np.dot(weights.transpose(), delta) * sp         # backprop to calculate error (delta) at layer - 1

    delta_b = delta
    delta_w = np.dot(delta, output.transpose())
    # print "backprop_1d_to_1d next_weights : ", next_weights
    # print "backprop_1d_to_1d delta : ", delta
    # print "backprop_1d_to_1d delta_w : ", delta_w.shape
    # print "backprop_1d_to_1d delta_b : ", delta_b.shape
    # log.debug("-> [backprop_1d_to_1d]  delta %s, delta_w : %s, delta_b : %s ",delta.shape, delta_w.shape, delta_b.shape)

    return delta_b, delta_w, delta

@numba.njit()
def backprop_1d_to_1d_final(loss_prime, output, z_vals):
    # log.debug("## delta.shape : %s", delta.shape)
    # log.debug("## weights.shape : %s", weights)
    # log.debug("## z_vals.shape : %s", z_vals.shape)

    sp = activation_prime(z_vals)
    delta = loss_prime * sp

    delta_b = delta
    delta_w = np.dot(delta, output.transpose())
    # print "backprop_1d_to_1d next_weights : ", next_weights
    # print "backprop_1d_to_1d delta : ", delta
    # print "backprop_1d_to_1d delta_w : ", delta_w.shape
    # print "backprop_1d_to_1d delta_b : ", delta_b.shape
    # log.debug("-> [backprop_1d_to_1d]  delta %s, delta_w : %s, delta_b : %s ",delta.shape, delta_w.shape, delta_b.shape)


    return delta_b, delta_w, delta

#fc to pool
@numba.njit()
def backprop_3d_to_1d(delta, weights, output, z_vals):
    # log.debug("## delta.shape : %s", delta.shape)
    # log.debug("## weights.shape : %s", weights.shape)
    # log.debug("## z_vals.shape : %s", z_vals.shape)

    # sp = activation_prime(z_vals)

    sp = activation_prime(z_vals)

    # print 'w,d,z_vals: ', prev_weights.shape, delta.shape, sp.shape, prev_activations.shape
    delta = np.dot(weights.transpose(), delta) * sp         # backprop to calculate error (delta) at layer - 1

    delta_b = delta
    depth, dim1, dim2 = output.shape
    output = output.reshape((1, depth * dim1 * dim2))
    delta_w = np.dot(delta, output)
    delta_w = delta_w.reshape((delta.shape[0], depth,dim1,dim2))

    # print "backprop_1d_to_1d next_weights : ", next_weights.shape
    # print "backprop_1d_to_3d delta : ", delta.shape
    # print "backprop_1d_to_3d delta_w : ", delta_w.shape
    # print "backprop_1d_to_3d delta_b : ", delta_b.shape
    # log.debug("-> [backprop_3d_to_1d]  delta %s, delta_w : %s, delta_b : %s ", delta.shape, delta_w.shape,delta_b.shape)

    return delta_b, delta_w, delta

#test(
@numba.njit()
def backprop_conv(delta, weights_shape, stride, output, z_vals):
    '''weights passed in are the ones between pooling and fc layer'''

    # log.debug("## delta.shape : %s", delta.shape)
    # log.debug("## weights.shape : %s", weights_shape)
    # log.debug("## stride : %s", stride)
    # log.debug("## output.shape : %s", output.shape)
    # log.debug("## prev_z_vals.shape : %s", prev_z_vals.shape)

    # print 'weight filter, delta shape', weight_filters.shape, delta.shape
    # print 'input shape', input_to_conv.shape

    num_filters, depth, filter_size, filter_size = weights_shape

    delta = activation_prime(z_vals) * delta

    delta_b = np.zeros((num_filters, 1))
    delta_w = np.zeros((weights_shape))  # you need to change the dims of weights

    x, y, z = delta.shape

    # print delta_w.shape, delta_b.shape, delta.shape
    total_deltas_per_layer = y * z
    # print 'total_deltas_per_layer', total_deltas_per_layer
    delta = delta.reshape((x, y * z))
    # log.debug("## delta.reshape : %s", delta.shape)
    # log.debug("## total_deltas_per_layer: %s", total_deltas_per_layer)
    # log.debug("## num_filters: %s", num_filters)
    # log.debug("## delta_w.shape : %s", delta_w.shape)
    # log.debug("## delta_b.shape : %s", delta_b.shape)
    z_vals = z_vals.reshape((z_vals.shape[0], z_vals.shape[1] * z_vals.shape[2]))

    #PARALEL WITH NUMBA
    # import time
    # start = time.time()
    delta, delta_b, delta_w = backprop_conv_loop(num_filters, total_deltas_per_layer, output, z_vals, filter_size, delta, delta_w, delta_b,stride)
    # end = time.time()
    # time = end - start
    # print "TIME : ", time

    delta = delta.reshape((x, y, z))

    # print "backprop_1d_to_1d next_weights : ", next_weights.shape
    # print "backprop_to_conv delta_w : ", delta_w.shape
    # print "backprop_to_conv delta_b : ", delta_b.shape
    # log.debug("-> [backprop_to_conv]  delta %s, delta_w : %s, delta_b : %s ", delta.shape, delta_w.shape,delta_b.shape)
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
    delta = np.dot(weights.transpose(), delta)         # versi awb without sp because there's no activation function
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
    # log.debug( "## delta.shape : %s", delta.shape)
    # log.debug( "## weights.shape : %s", weights_shape)
    # log.debug( "## stride : %s", stride)
    # log.debug( "## output.shape : %s", output.shape)
    # log.debug( "## prev_z_vals.shape : %s", prev_z_vals.shape)

    '''weights passed in are the ones between pooling and fc layer'''

    # print 'weight filter, delta shape', weight_filters.shape, delta.shape
    # print 'input shape', input_to_conv.shape
    num_filters, depth, filter_size, filter_size = weights_shape

    delta_b = np.zeros((num_filters, 1))
    delta_w = np.zeros((weights_shape))            # you need to change the dims of weights



    # print delta_w.shape, delta_b.shape, delta.shape
    total_deltas_per_layer = (delta.shape[1]) * (delta.shape[2])
    # print 'total_deltas_per_layer', total_deltas_per_layer
    delta = delta.reshape((delta.shape[0], delta.shape[1] * delta.shape[2]))

    # print "## delta.reshape : ", delta.shape
    # print "## total_deltas_per_layer: ", total_deltas_per_layer
    # print "## num_filters: ", num_filters

    #PARALLEL
    # import time
    # start = time.time()
    backprop_to_conv_loop(num_filters, total_deltas_per_layer, output, filter_size, delta, delta_w, delta_b, stride)
    # end = time.time()
    # time = end - start
    # print "TIME backprop_to_conv_loop: ",time


    # print "backprop_1d_to_1d next_weights : ", next_weights.shape
    # print "backprop_to_conv delta_w : ", delta_w.shape
    # print "backprop_to_conv delta_b : ", delta_b.shape
    # log.debug("-> [backprop_to_conv]  delta %s, delta_w : %s, delta_b : %s ", delta.shape, delta_w.shape,delta_b.shape)
    return delta_b, delta_w






