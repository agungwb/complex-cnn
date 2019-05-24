##FEED FORWARD OPTIMIZATION
import numba
# from numba import cuda
import numpy as np
from helper import *


# @numba.njit()
# def update_mini_batch_loop(layers, eta, nabla_w, nabla_b, batch_size, weight_index):
#     for ix, (layer_nabla_w, layer_nabla_b) in enumerate(zip(nabla_w, nabla_b)):
#         layer = layers[weight_index[ix]]
#         # print "type(layer_nabla_w) : ",layer_nabla_w
#         # print "type(layer_nabla_b) : ",layer_nabla_b
#         # print "type(layer_nabla_b) : ",layer_nabla_b
#         layer.weights -= eta * layer_nabla_w / batch_size
#         layer.biases -= eta * layer_nabla_b / batch_size



#FEED FORWARD
# @numba.njit(parallel=True)
@numba.njit()
def convole_loop(num_filters, z_values, input_neurons, width_in, weights, filter_size, stride, biases, output):
    act_length1d = output.shape[1]  # dim1 * dim2

    for j in numba.prange(num_filters):
        slide = 0
        row = 0

        for i in numba.prange(act_length1d):  # loop til the output array is filled up -> one dimensional (600)

            # ACTIVATIONS -> loop through each conv block horizontally

            # log.debug("z_values[j][i].shape : %s", type(z_values[j][i]))

            # z_values[j][i] = np.add(np.sum(np.multiply(input_neurons[:, row:filter_size + row, slide:filter_size + slide], weights[j])), biases[j])


            sub_matrix = input_neurons[:, row:filter_size + row, slide:filter_size + slide][0]
            weight = weights[j][0]
            # weight = rot180(weights[j][0]) #rotated

            input_sum = np.sum(np.multiply(sub_matrix, weight)) + biases[j]
            z_values[j][i] = input_sum[0]

            z = z_values[j][i]
            # output[j][i] = 1.0/(1.0 + np.exp(-z)) # activation function
            output[j][i] = activation(z) # activation function

            slide += stride

            if (filter_size + slide) - stride >= width_in:  # wrap indices at the end of each row
                slide = 0
                row += stride  # go to next row

    return z_values, output


@numba.njit()
def pool_loop(depth, pool_length1d, input_image, width_in, poolsize, max_indices, output):
    # for each filter map
    for j in numba.prange(depth):
        row = 0
        slide = 0
        for i in numba.prange(pool_length1d):
            toPool = input_image[j][row:poolsize[0] + row, slide:poolsize[0] + slide]


            #new algo
            max = toPool[0][0]
            index = list([0, 0])
            for r in numba.prange(poolsize[0]):
                for c in numba.prange(poolsize[0]):
                    if toPool[r, c] > max:
                        max = toPool[r, c]
                        index = list([r, c])

            output[j][i] = max
            index = list([index[0] + row, index[1] + slide])
            # print "---------------"
            # print "output[j][i] : ", output[j][i]
            # print "max : ", max
            # print "pos : ", pos
            #new algo end

            max_indices[j][i][0] = index[0]
            max_indices[j][i][1] = index[1]
            # print "max_indices[j][i] : ", max_indices[j][i]
            # sys.exit(0)

            slide += poolsize[1]

            # modify this if stride != filter for poolsize
            if slide >= width_in:
                slide = 0
                row += poolsize[1]


# BACKPROP
@numba.njit()
def backprop_conv_loop(num_filters, total_deltas_per_layer, output, z_vals, filter_size, delta, delta_w, delta_b, stride):
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

            # delta_w[j] += to_conv * delta[j][i]

            delta_w[j] += to_conv * delta[j][i] #versi sotoy awb
            delta_b[j] += delta[j][i]  # not fully sure, but im just summing up the bias deltas over the conv layer
            slide += stride

            if (slide + filter_size) - stride >= output.shape[2]:  # wrap indices at the end of each row
                slide = 0
                row += stride

    return delta, delta_b, delta_w

@numba.njit()
def backprop_pool_from_conv_loop(delta, weights, pool_output, stride, filter_size):
    # import time
    # start = time.time()
    x, y, z = pool_output.shape
    depth, dim1, dim2 = delta.shape

    # h = ((dim1-1) * stride) + filter_size - (2*padding)
    # w = ((dim2-1) * stride) + filter_size - (2*padding)

    delta_new = np.zeros((x, y * z))

    num_filters = x
    act_length1d = y * z

    for d in numba.prange(depth):
        # h_gap = (h - dim1) / 2
        # w_gap = (w - dim2) / 2

        h_gap = filter_size - 1
        w_gap = filter_size - 1

        height_in = dim1 + 2 * h_gap
        width_in = dim2 + 2 * w_gap

        # delta_padded_zero = np.zeros((height_in, width_in))
        # delta_padded_zero[h_gap:dim1 + h_gap, w_gap:dim2 + w_gap] = delta_temp[d]

        old_delta_padded_zero = delta_padded_zeros(height_in, width_in, h_gap, w_gap, dim1, dim2, delta[d])


        for j in numba.prange(num_filters):
            column = 0
            row = 0

            filter_rotated = rot180(weights[d, j])

            for i in numba.prange(act_length1d):

                # z = pool_output[j, row, column]
                # sigmoid = 1.0/(1.0 + np.exp(-z))
                # sp = sigmoid * (1-sigmoid)

                delta_new[j][i] += np.sum(np.multiply(old_delta_padded_zero[row:filter_size + row, column:filter_size + column], filter_rotated))

                # print "delta[",j,"][",i,"] : ",delta[j][i]

                # sys.exit(0)
                column += stride

                if (filter_size + column) - stride >= width_in:  # wrap indices at the end of each row

                    column = 0
                    row += stride  # go to next row

    return delta_new

    # end = time.time()
    # time = end - start
    # print "TIME : ", time

@numba.njit()
def backprop_pool_loop(input_from_conv, max_indices, poolsize, pool_output, delta):

    depth, height, width = input_from_conv.shape
    delta_new = np.zeros((depth, height, width))  # calc the delta on the conv layer

    for d in range(depth):    # depth is the same for conv + pool layer
        row = 0
        slide = 0
        for i in range(max_indices.shape[1]):
            toPool = input_from_conv[d][row:poolsize[0] + row, slide:poolsize[0] + slide]

            # calculate the new delta for the conv layer based on the max result + pooling input
            # print "pool_output[d][i] : ",pool_output[d][i]
            # print "delta[d][i] : ",delta[d][i]
            # print "toPool : ",toPool

            # deltas_from_pooling_old = max_prime(pool_output[d][i], delta[d][i], toPool)

            delta_pool = delta[d][i]
            res = pool_output[d][i]
            dim1, dim2 = toPool.shape
            #reshape
            # tile_to_pool = toPool.reshape((dim1 * dim2))
            tile_to_pool = np.zeros(dim1 * dim2)
            for r in range(dim1):
                for c in range(dim2):
                    tile_to_pool[r*dim1 + c] = toPool[r][c]

            new_delta = np.zeros((tile_to_pool.shape))

            for i in range(len(tile_to_pool)):
                num = tile_to_pool[i]
                if num < res:
                    new_delta[i] = 0
                else:
                    new_delta[i] = delta_pool

            deltas_from_pooling = new_delta.reshape((dim1, dim2))

            # print "---------------------"
            # print "deltas_from_pooling_old : ",deltas_from_pooling_old
            # print "deltas_from_pooling : ",deltas_from_pooling

            delta_new[d][row:poolsize[0] + row, slide:poolsize[0] + slide] = deltas_from_pooling

            slide += poolsize[1]
            if slide >= width:
                slide = 0
                row+= poolsize[1]
    return delta_new

@numba.njit(parallel=True)
def backprop_to_conv_loop(num_filters, total_deltas_per_layer, output, filter_size, delta, delta_w, delta_b, stride):
    for j in numba.prange(num_filters):
        slide = 0
        row = 0

        for i in numba.prange(total_deltas_per_layer):
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

