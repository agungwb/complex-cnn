from helper import *

import logging

import numpy as np
import numba

# try:
#     import cupy as np
# except ImportError:
#     import numpy as np

log = logging.getLogger("__backprop__")


# backpropagation
##############################################################

# delta_L = weights_L (dot) delta_L+1 .*  activation_prime(z_L)


def backprop_1d_to_1d(delta, weights, output, z_vals, final=False):
    log.debug("## delta.shape : %s", delta.shape)
    log.debug("## weights.shape : %s", weights)
    log.debug("## z_vals.shape : %s", z_vals.shape)

    if not final: # if final delta from loss function
        sp = activation_prime(z_vals)
        # print 'w,d,z_vals: ', prev_weights.shape, delta.shape, sp.shape, prev_activations.shape
        delta = np.dot(weights.transpose(), delta) * sp         # backprop to calculate error (delta) at layer - 1

    delta_b = delta
    delta_w = np.dot(delta, output.transpose())
    # print "backprop_1d_to_1d next_weights : ", next_weights
    # print "backprop_1d_to_1d delta : ", delta
    # print "backprop_1d_to_1d delta_w : ", delta_w.shape
    # print "backprop_1d_to_1d delta_b : ", delta_b.shape
    log.debug("-> [backprop_1d_to_1d]  delta %s, delta_w : %s, delta_b : %s ",delta.shape, delta_w.shape, delta_b.shape)


    return delta_b, delta_w, delta

#fc to pool
def backprop_3d_to_1d(delta, weights, output, z_vals):
    log.debug("## delta.shape : %s", delta.shape)
    log.debug("## weights.shape : %s", weights.shape)
    log.debug("## z_vals.shape : %s", z_vals.shape)

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
    log.debug("-> [backprop_3d_to_1d]  delta %s, delta_w : %s, delta_b : %s ", delta.shape, delta_w.shape,delta_b.shape)

    return delta_b, delta_w, delta

#test
def backprop_pool_to_conv(delta, weights_shape, stride, output, prev_z_vals):
    '''weights passed in are the ones between pooling and fc layer'''

    log.debug("## delta.shape : %s", delta.shape)
    log.debug("## weights.shape : %s", weights_shape)
    log.debug("## stride : %s", stride)
    log.debug("## output.shape : %s", output.shape)
    log.debug("## prev_z_vals.shape : %s", prev_z_vals.shape)

    # print 'weight filter, delta shape', weight_filters.shape, delta.shape
    # print 'input shape', input_to_conv.shape
    num_filters, depth, filter_size, filter_size = weights_shape

    delta_b = np.zeros((num_filters, 1))
    delta_w = np.zeros((weights_shape))  # you need to change the dims of weights

    # print delta_w.shape, delta_b.shape, delta.shape
    total_deltas_per_layer = (delta.shape[1]) * (delta.shape[2])
    # print 'total_deltas_per_layer', total_deltas_per_layer
    delta = delta.reshape((delta.shape[0], delta.shape[1] * delta.shape[2]))
    log.debug("## delta.reshape : %s", delta.shape)
    log.debug("## total_deltas_per_layer: %s", total_deltas_per_layer)
    log.debug("## num_filters: %s", num_filters)
    log.debug("## delta_w.shape : %s", delta_w.shape)
    log.debug("## delta_b.shape : %s", delta_b.shape)


    for j in range(num_filters):
        slide = 0
        row = 0

        for i in range(total_deltas_per_layer):
            to_conv = output[:, row:filter_size + row, slide:filter_size + slide]
            # print "## slide : ", slide
            # print "## filter_size : ", filter_size
            # print "## to_conv.shape : ", to_conv.shape
            # print "## delta_w[j].shape  : ", delta_w[j].shape
            #
            # sys.exit(0)

            delta_w[j] += to_conv * delta[j][i]
            delta_b[j] += delta[j][i]  # not fully sure, but im just summing up the bias deltas over the conv layer
            slide += stride

            if (slide + filter_size) - stride >= output.shape[2]:  # wrap indices at the end of each row
                slide = 0
                row += stride

    # print "backprop_1d_to_1d next_weights : ", next_weights.shape
    # print "backprop_to_conv delta_w : ", delta_w.shape
    # print "backprop_to_conv delta_b : ", delta_b.shape
    log.debug("-> [backprop_to_conv]  delta %s, delta_w : %s, delta_b : %s ", delta.shape, delta_w.shape,delta_b.shape)
    return delta_b, delta_w

def backprop_conv_to_pool(delta, weights, input_from_conv, max_indices, poolsize, pool_output, from_conv=False):
    # log.debug("## delta.shape : %s", delta.shape)
    # log.debug("## weights.shape : %s", weights.shape)
    # log.debug("## input_from_conv.shape : %s", input_from_conv.shape)
    # log.debug("## max_indices.shape : %s", max_indices.shape)
    # log.debug("## poolsize: %s", poolsize)
    # log.debug("## pool_output.shape: %s", pool_output.shape)

    # reshape the "z values" of the pool layer
    x,y,z = pool_output.shape
    a,b,c,d = weights.shape


    # same for the max index matrix
    max_indices = max_indices.reshape((x, y * z, 2))

    # backprop delta from fc to pool layer



    if not from_conv:
        weights = weights.reshape((a, b * c * d))
        pool_output = pool_output.reshape((x * y * z, 1))

        # sp = pool_output #versi awb sotoy, pooling gak pake activation
        sp = activation_prime(pool_output) #versi old
        # log.debug("## sp.shape : %s", sp.shape)
        # log.debug("## weights.transpose().shape : %s", weights.transpose().shape)
        delta = np.dot(weights.transpose(), delta) * sp         # backprop to calc delta on pooling layer
        # log.debug("## delta.shape (after) : %s", delta.shape)
        delta = delta.reshape((x, y * z))
    else:
        stride = 1
        filter_size = c
        padding = 0

        delta_temp = delta
        depth, dim1, dim2 = delta_temp.shape

        h = ((dim1-1) * stride) + filter_size - (2*padding)
        w = ((dim2-1) * stride) + filter_size - (2*padding)

        delta = np.zeros((x,y*z))

        num_filters = x
        act_length1d = y*z

        # print "## delta_temp.shape : ",delta_temp.shape
        # print "## delta.shape : ",delta.shape
        # print "## dfilter_size : ",filter_size
        # print "## num_filters : ",num_filters
        # print "## act_length1d : ",act_length1d

        ##function
        function_1(depth, filter_size, dim1, dim2, delta_temp, num_filters, weights, act_length1d, pool_output, delta,
                   stride)
        ##endfunction


    pool_output = pool_output.reshape((x, y * z))
    
    depth, height, width = input_from_conv.shape
    delta_new = np.zeros((depth, height, width)) # calc the delta on the conv layer

    # log.debug( "## pool_output.shape : %s", pool_output.shape)
    # log.debug( "## delta_new.shape : %s", delta_new.shape)

    for d in range(depth):    # depth is the same for conv + pool layer
        row = 0
        slide = 0
        for i in range(max_indices.shape[1]):
            toPool = input_from_conv[d][row:poolsize[0] + row, slide:poolsize[0] + slide]

            # calculate the new delta for the conv layer based on the max result + pooling input
            # print "pool_output[d][i] : ",pool_output[d][i]
            # print "delta[d][i] : ",delta[d][i]
            # print "toPool : ",toPool
            deltas_from_pooling = max_prime(pool_output[d][i], delta[d][i], toPool)

            # print "deltas_from_pooling : ",deltas_from_pooling
            # sys.exit(0)

            delta_new[d][row:poolsize[0] + row, slide:poolsize[0] + slide] = deltas_from_pooling

            slide += poolsize[1]
            if slide >= width:
                slide = 0
                row+= poolsize[1]



    # print "backprop_1d_to_1d next_weights : ", next_weights.shape
    # print "backprop_pool_to_conv : ", delta_new.shape
    # log.debug( "-> [backprop_conv_to_pool]  delta : %s", delta_new.shape)
    return delta_new

# @numba.jit(nopython=True, parallel=True)
def function_1(depth, filter_size, dim1, dim2, delta_temp, num_filters, weights, act_length1d, pool_output, delta, stride):
    for d in range(depth):
        # h_gap = (h - dim1) / 2
        # w_gap = (w - dim2) / 2

        h_gap = filter_size - 1
        w_gap = filter_size - 1

        height_in = dim1 + 2 * h_gap
        width_in = dim2 + 2 * w_gap

        delta_padded_zero = np.zeros((height_in, width_in))
        delta_padded_zero[h_gap:dim1 + h_gap, w_gap:dim2 + w_gap] = delta_temp[d]

        # print "delta_pedded_zero.shape : ",delta_padded_zero.shape
        # print "h_gap : ",h_gap
        # print "w_gap : ",w_gap

        for j in range(num_filters):
            column = 0
            row = 0

            # print "d : ",d
            # print "i : ",i
            # print "j : ",j
            # print "weights.shape : ",weights.shape
            # print "weights[d,j] : ", weights[d,j].shape
            filter_rotated = np.rot90(np.rot90(weights[d, j]))

            for i in range(act_length1d):

                # print "depth : ", depth
                # print "dim1 : ", dim1
                # print "dim2 : ", dim2
                # print "h : ", h
                # print "w : ", w
                # print "delta_padded_zero.shape : ", delta_padded_zero.shape
                # print "filter.shape : ", filter.shape

                # ACTIVATIONS -> loop through each conv block horizontally
                # sp = activation_prime(pool_output[i])
                # print "---"
                # print "i : ", i
                # print "j : ", j
                # print "row : ",row
                # print "filter_size + row : ", filter_size + row
                # print "filter_size : ",filter_size
                # print "column : ",column
                # print "filter_size + column", filter_size + column

                #
                # temp = delta_padded_zero[row:filter_size + row, column:filter_size + column]
                # print "temp.shape : ",temp.shape

                sp = activation_prime(pool_output[j, row, column])

                # if isinstance(pool_output[j, row, column], (list,)):
                #     activation_prime_parallel = numba.jit("f8[:](f8[:])")(activation_prime())
                # else:
                #     activation_prime_parallel = numba.jit("f8(f8)")(activation_prime)
                #
                # sp = activation_prime_parallel(pool_output[j, row, column])

                delta[j][i] += np.sum(
                    delta_padded_zero[row:filter_size + row, column:filter_size + column] * filter_rotated) * sp

                # print "delta[",j,"][",i,"] : ",delta[j][i]

                # sys.exit(0)
                column += stride

                if (filter_size + column) - stride >= width_in:  # wrap indices at the end of each row
                    # print "-------"
                    # print "i : ",i
                    # print "filter_size : ",filter_size
                    # print "row : ", row
                    # print "column : ",column
                    # print "stride : ",stride
                    # print "act_length1d : ",act_length1d
                    column = 0
                    row += stride  # go to next row


def backprop_to_conv(delta, weights_shape, stride, output, prev_z_vals):
    log.debug( "## delta.shape : %s", delta.shape)
    log.debug( "## weights.shape : %s", weights_shape)
    log.debug( "## stride : %s", stride)
    log.debug( "## output.shape : %s", output.shape)
    log.debug( "## prev_z_vals.shape : %s", prev_z_vals.shape)

    '''weights passed in are the ones between pooling and fc layer'''

    delta = delta

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

    for j in range(num_filters):
        slide = 0
        row = 0

        for i in range(total_deltas_per_layer):
            to_conv = output[:, row:filter_size + row, slide:filter_size + slide]
            # print "## slide : ", slide
            # print "## filter_size : ", filter_size
            # print "## to_conv.shape : ", to_conv.shape
            # print "## delta_w[j].shape  : ", delta_w[j].shape
            # sys.exit(0)
            delta_w[j] += to_conv * delta[j][i]
            delta_b[j] += delta[j][i]       # not fully sure, but im just summing up the bias deltas over the conv layer
            slide += stride

            if (slide + filter_size)-stride >= output.shape[2]:    # wrap indices at the end of each row
                slide = 0
                row+=stride

    # print "backprop_1d_to_1d next_weights : ", next_weights.shape
    # print "backprop_to_conv delta_w : ", delta_w.shape
    # print "backprop_to_conv delta_b : ", delta_b.shape
    log.debug("-> [backprop_to_conv]  delta %s, delta_w : %s, delta_b : %s ", delta.shape, delta_w.shape,delta_b.shape)
    return delta_b, delta_w




