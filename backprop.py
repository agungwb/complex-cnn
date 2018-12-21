import numpy as np
from helper import *


# backpropagation
##############################################################

def backprop_1d_to_1d(next_delta, next_weights, prev_output, z_vals, final=False):
    delta = next_delta

    if not final: # reset delta
        sp = activation_prime(z_vals)
        # print 'w,d,z_vals: ', prev_weights.shape, delta.shape, sp.shape, prev_activations.shape
        delta = np.dot(next_weights.transpose(), delta) * sp         # backprop to calculate error (delta) at layer - 1

    delta_b = delta
    delta_w = np.dot(delta, prev_output.transpose())
    # print "backprop_1d_to_1d next_weights : ", next_weights
    # print "backprop_1d_to_1d delta : ", delta
    # print "backprop_1d_to_1d delta_w : ", delta_w.shape
    # print "backprop_1d_to_1d delta_b : ", delta_b.shape
    return delta_b, delta_w, delta

#fc to pool
def backprop_1d_to_3d(next_delta, next_weights, prev_output, z_vals):
    delta = next_delta

    sp = activation_prime(z_vals)
    # print 'w,d,z_vals: ', prev_weights.shape, delta.shape, sp.shape, prev_activations.shape
    delta = np.dot(next_weights.transpose(), delta) * sp         # backprop to calculate error (delta) at layer - 1

    delta_b = delta
    depth, dim1, dim2 = prev_output.shape
    prev_output = prev_output.reshape((1, depth * dim1 * dim2))
    delta_w = np.dot(delta, prev_output)
    delta_w = delta_w.reshape((delta.shape[0], depth,dim1,dim2))

    # print "backprop_1d_to_1d next_weights : ", next_weights.shape
    # print "backprop_1d_to_3d delta : ", delta.shape
    # print "backprop_1d_to_3d delta_w : ", delta_w.shape
    # print "backprop_1d_to_3d delta_b : ", delta_b.shape
    return delta_b, delta_w, delta

#test
def backprop_conv_to_pool(next_delta, next_weights, prev_output, z_vals):
    delta = next_delta

    sp = activation_prime(z_vals)
    # print 'w,d,z_vals: ', prev_weights.shape, delta.shape, sp.shape, prev_activations.shape
    delta = np.dot(next_weights.transpose(), delta) * sp         # backprop to calculate error (delta) at layer - 1

    delta_b = delta
    depth, dim1, dim2 = prev_output.shape
    prev_output = prev_output.reshape((1, depth * dim1 * dim2))
    delta_w = np.dot(delta, prev_output)
    delta_w = delta_w.reshape((delta.shape[0], depth,dim1,dim2))

    # print "backprop_1d_to_1d next_weights : ", next_weights.shape
    # print "backprop_1d_to_3d delta : ", delta.shape
    # print "backprop_1d_to_3d delta_w : ", delta_w.shape
    # print "backprop_1d_to_3d delta_b : ", delta_b.shape
    return delta_b, delta_w, delta

    
def backprop_pool_to_conv(next_delta, next_weights, input_from_conv, max_indices, poolsize, pool_output):
    delta = next_delta

    # reshape the "z values" of the pool layer
    x,y,z = pool_output.shape
    a,b,c,d = next_weights.shape
    next_weights = next_weights.reshape((a, b * c * d))
    pool_output= pool_output.reshape((x * y * z,1))

    # same for the max index matrix
    max_indices = max_indices.reshape((x, y * z, 2))

    # backprop delta from fc to pool layer
    sp = activation_prime(pool_output)
    delta = np.dot(next_weights.transpose(), delta) * sp         # backprop to calc delta on pooling layer
    delta = delta.reshape((x,y*z))
    pool_output = pool_output.reshape((x, y * z))
    
    depth, height, width = input_from_conv.shape
    delta_new = np.zeros((depth, height, width)) # calc the delta on the conv layer

    for d in range(depth):    # depth is the same for conv + pool layer
        row = 0
        slide = 0
        for i in range(max_indices.shape[1]):
            toPool = input_from_conv[d][row:poolsize[0] + row, slide:poolsize[0] + slide]

            # calculate the new delta for the conv layer based on the max result + pooling input
            deltas_from_pooling = max_prime(pool_output[d][i], delta[d][i], toPool)
            delta_new[d][row:poolsize[0] + row, slide:poolsize[0] + slide] = deltas_from_pooling

            slide += poolsize[1]
            if slide >= width:
                slide = 0
                row+= poolsize[1]

    # print "backprop_1d_to_1d next_weights : ", next_weights.shape
    # print "backprop_pool_to_conv : ", delta_new.shape
    return delta_new

def backprop_to_conv(next_delta, next_weights, stride, prev_output, prev_z_vals):
    '''weights passed in are the ones between pooling and fc layer'''

    delta = next_delta

    # print 'weight filter, delta shape', weight_filters.shape, delta.shape
    # print 'input shape', input_to_conv.shape
    num_filters, depth, filter_size, filter_size = next_weights.shape

    delta_b = np.zeros((num_filters, 1))
    delta_w = np.zeros((next_weights.shape))            # you need to change the dims of weights



    # print delta_w.shape, delta_b.shape, delta.shape
    total_deltas_per_layer = (delta.shape[1]) * (delta.shape[2])
    # print 'total_deltas_per_layer', total_deltas_per_layer
    delta = delta.reshape((delta.shape[0], delta.shape[1] * delta.shape[2]))

    for j in range(num_filters):
        slide = 0
        row = 0

        for i in range(total_deltas_per_layer):
            to_conv = prev_output[:, row:filter_size + row, slide:filter_size + slide]
            delta_w[j] += to_conv * delta[j][i]
            delta_b[j] += delta[j][i]       # not fully sure, but im just summing up the bias deltas over the conv layer
            slide += stride

            if (slide + filter_size)-stride >= prev_output.shape[2]:    # wrap indices at the end of each row
                slide = 0
                row+=stride

    # print "backprop_1d_to_1d next_weights : ", next_weights.shape
    # print "backprop_to_conv delta_w : ", delta_w.shape
    # print "backprop_to_conv delta_b : ", delta_b.shape
    return delta_b, delta_w




