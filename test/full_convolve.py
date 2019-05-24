import numpy as np

def full_convolve(delta, weight):

    filter_rotated = rot180(weight)

    for i in range(act_length1d):

        z = pool_output[j, row, column]
        # sigmoid = 1.0/(1.0 + np.exp(-z))
        # sp = sigmoid * (1-sigmoid)

        delta[j][i] += np.sum(np.multiply(delta_padded_zero[row:filter_size + row, column:filter_size + column], filter_rotated))



def rot180(a):
    row, col = a.shape
    temp = np.zeros((row, col), dtype=np.float64)
    for x in range(col, 0, -1):
        for y in range(row, 0, -1):
            # print("x, y : %s, %s", x, y)
            # print("5-x, 5-y : %s, %s", col-x, row-y)
            temp[x - 1][y - 1] = a[row - x][col - y]
    return temp