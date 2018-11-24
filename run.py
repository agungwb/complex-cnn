import mnist_loader
from cnn import *
import sys
from backprop import *

import collections


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


ETA = 1.5
EPOCHS = 1 #default 5
INPUT_SHAPE = (28*28)     # for mnist
BATCH_SIZE = 50  #defalut 10
LMBDA = 0.1

# import ipdb; ipdb.set_trace()
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()


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
# training_data = (training_data, label)
x,y = training_data[0][0].shape
input_shape = (1,x,y)
print 'shape of input data: ', input_shape

net = Model(input_shape,
            layer_config = [
                {'conv_layer': {
                    'filter_size' : 5,
                    'stride' : 1,
                    'num_filters' : 20}},
                {'pool_layer': {
                    'poolsize' : (2,2)}},
                {'fc_layer': {
                    'num_output' : 30}},
                {'final_layer': {
                    'num_classes' : 10}}
            ])


# print(np.shape(training_data))
# print(np.shape(validation_data))
# print(np.shape(test_data))
# print(training_data[0][0])
# print(training_data[0][1].shape)

# print(training_data[0][0])
# sys.exit(0)

# net.gradient_descent(training_data[0:100], BATCH_SIZE, ETA, EPOCHS, LMBDA, test_data = test_data[:20])
net.gradient_descent(training_data[0:100], BATCH_SIZE, ETA, EPOCHS, LMBDA, test_data = test_data[:20])
