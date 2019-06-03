import mnist_loader
import sys
import logging

from backprop import *
import mammogram_loader
import hanacaraka_loader
import dummy_loader
import number_loader
import numba

import numpy as np

# try:
#     import cupy as np
#     print "<<< Backend powered by CuPy >>>"
# except ImportError:
#     import numpy as np
#     print "<<< Backend powered by NumPy >>>"


if len(sys.argv) < 2:
    print("miss an argument: cnn [test] | ccnn [test]")
    print("system terminated")
    print(0)

if sys.argv[1] != 'cnn' \
        and sys.argv[1] != 'ccnn' \
        and sys.argv[1] != 'mnist' \
        and sys.argv[1] != 'hanacaraka' \
        and sys.argv[1] != 'hanacaraka-complex' \
        and sys.argv[1] != 'number' \
        and sys.argv[1] != 'number-complex' \
        and sys.argv[1] != 'dummy':
    print("wrong argument: cnn [test] | ccnn [test]")
    print("system terminated")
    print(0)

if sys.argv[1] == 'ccnn':
    from ccnn import *
else:
    from cnn import *



######################### TEST IMAGE ##########################
import scipy
from scipy import ndimage, misc
import matplotlib.pyplot as plt

# im = scipy.ndimage.imread('images/cat.jpg', flatten=True)
# a = im.shape[0]
# b= im.shape[1]
# cat = scipy.misc.imresize(im, (a/40,b/40), interp='bilinear', mode=None)
# # normalize
# cat = 1.0 - cat/255.0

######################### TEST IMAGE ##########################


ETA = 0.1 #learning-rate (maybe)
EPOCHS = 10 #default 5
WIDTH = 36
HEIGHT = 36
CHANNEL = 1
INPUT_SHAPE = (HEIGHT*WIDTH)     # for mnist
BATCH_SIZE = 32  #defalut 10
LMBDA = 0.1
OUTPUT = 1

# import ipdb; ipdb.set_trace()
# training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

if sys.argv[1] == 'ccnn':
    WIDTH = 18
    HEIGHT = 18
    OUTPUT = 1
    # training_data, validation_data, test_data = mammogram_loader.load_data_dtcwt(sys.argv[2] if len(sys.argv) > 2 else 'main')
    training_data, test_data = mammogram_loader.load_data_dtcwt(sys.argv[2] if len(sys.argv) > 2 else 'main')
    if (len(sys.argv) > 2 and sys.argv[2] == 'test'):
        BATCH_SIZE = 10  # defalut 10
        EPOCH = 1
        logging.basicConfig(level=logging.DEBUG)
        log = logging.getLogger("__run__")
    else:
        logging.basicConfig(level=logging.INFO)
        log = logging.getLogger("__run__")
elif sys.argv[1] == 'cnn':
    WIDTH = 36
    HEIGHT = 36
    OUTPUT = 1
    # training_data, validation_data, test_data = mammogram_loader.load_data(sys.argv[2] if len(sys.argv) > 2 else 'main')
    training_data, test_data = mammogram_loader.load_data(sys.argv[2] if len(sys.argv) > 2 else 'main')
    if (len(sys.argv) > 2 and sys.argv[2] == 'test'):
        BATCH_SIZE = 10  # defalut 10
        EPOCH = 1
        logging.basicConfig(level=logging.DEBUG)
        log = logging.getLogger("__run__")
    else:
        logging.basicConfig(level=logging.INFO)
        log = logging.getLogger("__run__")

elif sys.argv[1] == 'mnist':
    WIDTH = 28
    HEIGHT = 28
    OUTPUT = 10
    EPOCHS = 10
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger("__run__")

elif sys.argv[1] == 'hanacaraka':
    WIDTH = 78
    HEIGHT = 60
    OUTPUT = 1
    EPOCHS = 10
    training_data, test_data = hanacaraka_loader.load_data()
    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger("__run__")
elif sys.argv[1] == 'hanacaraka-complex':
    WIDTH = 78
    HEIGHT = 60
    OUTPUT = 1
    EPOCHS = 1
    training_data, test_data = hanacaraka_loader.load_data_dtcwt()
    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger("__run__")
elif sys.argv[1] == 'number':
    WIDTH = 28
    HEIGHT = 28
    OUTPUT = 1
    EPOCHS = 50
    training_data, test_data = number_loader.load_data()
    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger("__run__")
elif sys.argv[1] == 'dummy':
    WIDTH = 14
    HEIGHT = 14
    OUTPUT = 1
    EPOCHS = 10
    BATCH_SIZE = 1
    training_data, test_data = dummy_loader.load_data(max_range=5, dimension=(HEIGHT, WIDTH))
    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger("__run__")
else:
    print("no matched mode")
    print("system terminated")
    sys.exit(0)



'''
Args:
     Convolutional Layer: filter_size, stride, padding, num_filters
     Pooling Layer: poolsize
     Fully Connected Layer: num_output, classify = True/False, num_classes (if classify True)
     Gradient Descent: training data, batch_size, eta, num_epochs, lambda, test_data
'''

# training_data = cat.reshape((1,43,64))
# input_shape = training_data.shape
# label = np.asarray(([1,0])).reshape((2,1))

log.info("training_data[0][0] : %s",training_data[0][0].shape) #training_data = (n_data, label(input, output))
x,y = training_data[0][0].shape
input_shape = (1,x,y)
log.info('shape of input data: %s', input_shape)
log.info('len(training_data) : %s', len(training_data))
# print 'len(validation_data) : ', len(validation_data)
log.info('len(test_data) : %s', len(test_data))

#activation: 1. sigmoid, 2.tanh, 3.relu
#loss_function: 1. quadratic, 2. binary_cross_entropy

net = Model(input_shape,
            layer_config = [
                {'conv_layer':
                    {
                        'filter_size': 5,
                        'stride': 1,
                        'num_filters': 20,
                        'activation': 1
                    }
                },
                {'pool_layer':
                    {
                        'poolsize': (2,2)
                    }
                },
                {'conv_layer':
                    {
                        'filter_size': 3,
                        'stride': 1,
                        'num_filters': 50,
                        'activation': 1
                    }
                },
                {'pool_layer':
                    {
                        'poolsize': (2,2)
                    }
                },
                {'fc_layer':
                    {
                        'num_output': 30,
                        'activation': 1
                    }
                },
                {'final_layer':
                    {
                        'num_classes': OUTPUT,
                        'activation': 1,
                        'loss_function': 1
                    }
                }
            ])

# print(np.shape(training_data))
# print(np.shape(validation_data))
# print(np.shape(test_data))
# print(training_data[0][0])
# print(training_data[0][1].shape)
# print(training_data[0][0])
# sys.exit(0)

# net.gradient_descent(training_data[0:100], BATCH_SIZE, ETA, EPOCHS, LMBDA, test_data = test_data[:20])
# net.gradient_descent(training_data[:100], BATCH_SIZE, ETA, EPOCHS, LMBDA, test_data = test_data[:100])
start = time.time()
net.gradient_descent(training_data, BATCH_SIZE, ETA, EPOCHS, OUTPUT, LMBDA, test_data = test_data)
end = time.time()
time = end - start
log.info("Time : {0}".format(time))