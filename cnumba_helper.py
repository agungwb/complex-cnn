##FEED FORWARD OPTIMIZATION
import numba
# from numba import cuda
import numpy as np
from chelper import *


#FEED FORWARD
@numba.njit(parallel=True)
def convole_loop(num_filters, act_length1d, z_values, input_neurons, width_in, weights, filter_size, stride, biases, output):
    for j in numba.prange(num_filters):
        slide = 0
        row = 0

        for i in numba.prange(act_length1d):  # loop til the output array is filled up -> one dimensional (600)
            input_sum = np.sum(np.multiply(input_neurons[:, row:filter_size + row, slide:filter_size + slide], weights[j])) + biases[j]
            z_values[j][i] = input_sum[0]

            output[j][i] = activation(z_values[j][i])

            slide += stride

            if (filter_size + slide) - stride >= width_in:  # wrap indices at the end of each row
                slide = 0
                row += stride  # go to next row

@numba.njit()
def pool_loop(depth, pool_length1d, input_image, width_in, poolsize, max_indices, output):
    # for each filter map
    for j in numba.prange(depth):
        row = 0
        slide = 0
        for i in numba.prange(pool_length1d):
            toPool = input_image[j][row:poolsize[0] + row, slide:poolsize[0] + slide]
            max = toPool[0][0]
            index = list([0, 0])
            for r in numba.prange(poolsize[0]):
                for c in numba.prange(poolsize[0]):
                    m = np.abs(toPool[r,c])
                    m_max = np.abs(max)
                    if m > m_max:
                        max = toPool[r, c]
                        index = list([r, c])

            output[j][i] = max
            index = list([index[0] + row, index[1] + slide])

            max_indices[j][i][0] = index[0]
            max_indices[j][i][1] = index[1]

            slide += poolsize[1]

            # modify this if stride != filter for poolsize
            if slide >= width_in:
                slide = 0
                row += poolsize[1]


# BACKPROP
@numba.njit()
def backprop_pool_to_conv_loop(num_filters, total_deltas_per_layer, output, filter_size, delta, delta_w, delta_b, stride):
    for j in range(num_filters):
        slide = 0
        row = 0

        for i in range(total_deltas_per_layer):
            to_conv = output[:, row:filter_size + row, slide:filter_size + slide]

            delta_w[j] += np.multiply(to_conv, delta[j][i])
            delta_b[j] += delta[j][i]  # not fully sure, but im just summing up the bias deltas over the conv layer
            slide += stride

            if (slide + filter_size) - stride >= output.shape[2]:  # wrap indices at the end of each row
                slide = 0
                row += stride

@numba.njit()
def backprop_conv_to_pool_loop(depth, filter_size, dim1, dim2, delta_temp, num_filters, weights, act_length1d, pool_output, delta, stride):

    for d in numba.prange(depth):

        h_gap = filter_size - 1
        w_gap = filter_size - 1

        height_in = dim1 + 2 * h_gap
        width_in = dim2 + 2 * w_gap

        delta_padded_zero = delta_padded_zeros_complex(height_in, width_in, h_gap, w_gap, dim1, dim2, delta_temp[d])

        for j in numba.prange(num_filters):
            column = 0
            row = 0

            filter_rotated = rot180_complex(weights[d, j])

            for i in numba.prange(act_length1d):
                sp = activation_prime(pool_output[j, row, column])

                delta[j][i] += np.multiply(np.sum(np.multiply(delta_padded_zero[row:filter_size + row, column:filter_size + column], filter_rotated)),sp)

                column += stride

                if (filter_size + column) - stride >= width_in:  # wrap indices at the end of each row
                    column = 0
                    row += stride  # go to next row

    # end = time.time()
    # time = end - start
    # print "TIME : ", time

@numba.njit()
def backprop_conv_to_pool_loop1(depth, max_indices, input_from_conv, poolsize, pool_output, delta, width, delta_new):
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
            tile_to_pool = np.zeros((dim1 * dim2)) + 0j

            for r in range(dim1):
                for c in range(dim2):
                    tile_to_pool[r*dim1 + c] = toPool[r][c]

            new_delta = np.zeros((tile_to_pool.shape)) + 0j

            for i in range(len(tile_to_pool)):
                num = tile_to_pool[i]
                m_num = np.abs(num)
                m_res = np.abs(res)
                if m_num < m_res:
                    new_delta[i] = 0j
                else:
                    new_delta[i] = delta_pool

            deltas_from_pooling = new_delta.reshape((dim1, dim2))

            # print "---------------------"
            # print "deltas_from_pooling_old : ",deltas_from_pooling_old

            delta_new[d][row:poolsize[0] + row, slide:poolsize[0] + slide] = deltas_from_pooling

            slide += poolsize[1]
            if slide >= width:
                slide = 0
                row+= poolsize[1]


@numba.njit(parallel=True)
def backprop_to_conv_loop(num_filters, total_deltas_per_layer, output, filter_size, delta, delta_w, delta_b, stride):
    for j in numba.prange(num_filters):
        slide = 0
        row = 0

        for i in numba.prange(total_deltas_per_layer):
            to_conv = output[:, row:filter_size + row, slide:filter_size + slide]
            
            delta_w[j] += to_conv * delta[j][i]
            delta_b[j] += delta[j][i]       # not fully sure, but im just summing up the bias deltas over the conv layer
            slide += stride

            if (slide + filter_size)-stride >= output.shape[2]:    # wrap indices at the end of each row
                slide = 0
                row+=stride