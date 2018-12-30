# This is the basic idea behind the architecture

import random
import math
import math
import time
import sys
import scipy
from scipy import ndimage, misc
import matplotlib.pyplot as plt
from helper import *

from backprop import *
import logging

try:
    import cupy as np
except ImportError:
    import numpy as np

log = logging.getLogger("__cnn__")

''' RECEPTIVE FIELD - WEIGHTS aka FILTER ->
initialize filters in a way that corresponds to the depth of the image.
If the input image is of channel 3 (RGB) then each of your weight vector is n*n*3.
PARAMETERS you'll need: NUM_FILTERS (num of filters), STRIDE (slide filter by), ZERO-PADDING(to control the spatial size of the output volumes). Use (Inputs-FilterSize + 2*Padding)/Stride + 1 to calculate your output volume and to decide your hyperparameters'''


class ConvLayer(object):

    def __init__(self, input_shape, filter_size, stride, num_filters, padding=0):
        self.depth, self.height_in, self.width_in = input_shape
        self.filter_size = filter_size
        self.stride = stride
        self.padding = padding
        self.num_filters = num_filters
        # self.num_filters = num_filters * self.depth

        self.weights = np.random.randn(self.num_filters, self.depth, self.filter_size,
                                       self.filter_size)  # filter * depth * filter_size * filter_size
        self.biases = np.random.rand(self.num_filters, 1)  # filter * 1

        # np.random.randn generate random from normal distribution
        # np.random.rand generate random from [0..1]
        # random with complex value np.random.rand(4).view(np.complex128)

        # output image height after convolution ((h - filter_size) / stride)) + 1
        self.output_dim1 = (self.height_in - self.filter_size + 2 * self.padding) / self.stride + 1
        # output image width after confolution ((w - filter_size) / stride)) + 1
        self.output_dim2 = (self.width_in - self.filter_size + 2 * self.padding) / self.stride + 1

        # output convolution layer (before activation function) (num_filters, output_dim1, output_dim2)
        self.z_values = np.zeros((self.num_filters, self.output_dim1, self.output_dim2))

        # output convolution layer (after activation function) (num_filters, output_dim1, output_dim2)
        self.output = np.zeros((self.num_filters, self.output_dim1, self.output_dim2))

    def convolve(self, input_neurons):
        '''
        Pass in the actual input data and do the convolution.
        Returns: sigmoid activation matrix after convolution
        '''

        # roll out activations
        # print "input_neurons.shape : ",input_neurons.shape
        self.z_values = self.z_values.reshape((self.num_filters, self.output_dim1 * self.output_dim2))
        self.output = self.output.reshape((self.num_filters, self.output_dim1 * self.output_dim2))

        act_length1d = self.output.shape[1]  # dim1 * dim2

        for j in range(self.num_filters):
            slide = 0
            row = 0

            for i in range(act_length1d):  # loop til the output array is filled up -> one dimensional (600)

                # ACTIVATIONS -> loop through each conv block horizontally
                self.z_values[j][i] = np.add(np.sum(np.multiply(input_neurons[:, row:self.filter_size + row, slide:self.filter_size + slide], self.weights[j])), self.biases[j])
                self.output[j][i] = activation(self.z_values[j][i])  # activation function

                # print "input_neurons sub : ",input_neurons[:, row:self.filter_size + row, slide:self.filter_size + slide].shape
                # print "self.weights[j].shape : ",self.weights[j].shape
                # print "self.z_values[j][i] : ",self.z_values[j][i]
                # print "self.output[j][i] : ",self.output[j][i]
                # sys.exit(0)
                slide += self.stride

                if (self.filter_size + slide) - self.stride >= self.width_in:  # wrap indices at the end of each row
                    slide = 0
                    row += self.stride  # go to next row

        self.z_values = self.output.reshape((self.num_filters, self.output_dim1, self.output_dim2))
        self.output = self.output.reshape((self.num_filters, self.output_dim1, self.output_dim2))


class PoolingLayer(object):

    def __init__(self, input_shape, poolsize=(2, 2)):
        '''
        width_in and height_in are the dimensions of the input image
        poolsize is treated as a tuple of filter and stride -> it should work with overlapping pooling
        '''
        self.depth, self.height_in, self.width_in = input_shape
        self.poolsize = poolsize
        self.height_out = (self.height_in - self.poolsize[0]) / self.poolsize[1] + 1
        self.width_out = (self.width_in - self.poolsize[0]) / self.poolsize[1] + 1  # num of output neurons

        # print "self.height_in : ",self.height_in
        # print "self.poolsize[0] : ",self.poolsize[0]
        # print "self.poolsize[1] : ",self.poolsize[1]
        # print "self.height_out : ",self.height_out
        # print "self.width_out : ",self.width_out

        self.output = np.empty((self.depth, self.height_out, self.width_out))
        self.max_indices = np.empty((self.depth, self.height_out, self.width_out, 2))

    def pool(self, input_image):

        self.pool_length1d = self.height_out * self.width_out

        self.output = self.output.reshape((self.depth, self.pool_length1d))
        self.max_indices = self.max_indices.reshape(
            (self.depth, self.pool_length1d, 2))  # store index of max output come from

        # for each filter map
        for j in range(self.depth):
            row = 0
            slide = 0
            for i in range(self.pool_length1d):
                toPool = input_image[j][row:self.poolsize[0] + row, slide:self.poolsize[0] + slide]

                self.output[j][i] = np.amax(toPool)  # calculate the max activation
                # print ("toPool : ", toPool)
                # print ("np.max(toPool) : ", np.max(toPool))
                # print ("np.amax(toPool) : ", np.amax(toPool))
                # print ("p.where(np.max(toPool) == toPool) : ", np.where(np.max(toPool) == toPool))
                index = zip(*np.where(np.max(
                    toPool) == toPool))  # HERE IT IS save the index of the max, np.where return index of array if condition meets
                # print "index before : ", index
                if len(index) > 1:  # if there is more than one maximum value
                    index = [index[0]]
                # print "index after : ", index
                index = index[0][0] + row, index[0][1] + slide
                self.max_indices[j][i] = index

                slide += self.poolsize[1]

                # modify this if stride != filter for poolsize
                if slide >= self.width_in:
                    slide = 0
                    row += self.poolsize[1]

        self.output = self.output.reshape((self.depth, self.height_out, self.width_out))
        self.max_indices = self.max_indices.reshape((self.depth, self.height_out, self.width_out, 2))


class Layer(object):

    def __init__(self, input_shape, num_output):
        self.output = np.ones((num_output, 1))
        self.z_values = np.ones((num_output, 1))


class FullyConnectedLayer(Layer):
    '''
    Calculates outputs on the fully connected layer then forwardpasses to the final output -> classes
    '''

    def __init__(self, input_shape, num_output):
        super(FullyConnectedLayer, self).__init__(input_shape, num_output)
        self.depth, self.height_in, self.width_in = input_shape
        self.num_output = num_output

        self.weights = np.random.randn(self.num_output, self.depth, self.height_in, self.width_in)
        self.biases = np.random.randn(self.num_output, 1)

    def feedforward(self, a):
        '''
        forwardpropagates through the FC layer to the final output layer
        '''
        # roll out the dimensions
        self.weights = self.weights.reshape((self.num_output, self.depth * self.height_in * self.width_in))
        a = a.reshape((self.depth * self.height_in * self.width_in, 1))

        # this is shape of (num_outputs, 1)
        self.z_values = np.dot(self.weights, a) + self.biases
        self.output = activation(self.z_values)

        # print "self.z_values.shape : ", self.z_values.shape
        # print "self.output.shape : ", self.output.shape
        # print "self.weights.shape : ", self.weights.shape

        self.weights = self.weights.reshape((self.num_output, self.depth, self.height_in, self.width_in))

        # print "self.weights.shape2 : ", self.weights.shape


class ClassifyLayer(Layer):
    def __init__(self, num_inputs, num_classes):
        super(ClassifyLayer, self).__init__(num_inputs, num_classes)
        num_inputs, col = num_inputs
        self.num_classes = num_classes
        self.weights = np.random.randn(self.num_classes, num_inputs)
        # print "self.weights.shape : ", self.weights.shape
        self.biases = np.random.randn(self.num_classes, 1)

    def classify(self, x):
        self.z_values = np.dot(self.weights, x) + self.biases
        self.output = activation(self.z_values)
        # print "x : ", x
        # print "w : ", self.weights
        # print "z : ", self.z_values
        # print "o : ", self.output
        # print "self.z_values.shape : ", self.z_values.shape
        # print "self.output.shape : ", self.output.shape


class Model(object):
    layer_type_map = {
        'fc_layer': FullyConnectedLayer,
        'final_layer': ClassifyLayer,
        'conv_layer': ConvLayer,
        'pool_layer': PoolingLayer
    }

    def __init__(self, input_shape, layer_config):
        '''
        :param layer_config: list of dicts, outer key is
        Valid Layer Types:
        Convolutional Layer: shape of input, filter_size, stride, padding, num_filters
        Pooling Layer: shape of input(depth, height_in, width_in), poolsize
        Fully Connected Layer: shape_of_input, num_output, classify = True/False, num_classes (if classify True)
        Gradient Descent: training data, batch_size, eta, num_epochs, lambda, test_data
        '''

        self.input_shape = input_shape
        self._initialize_layers(layer_config)  # initiate layers based on layer_config
        self.layer_weight_shapes = [l.weights.shape for l in self.layers if not isinstance(l, PoolingLayer)]
        self.layer_biases_shapes = [l.biases.shape for l in self.layers if not isinstance(l, PoolingLayer)]

        # print "layer.shape : ", self.layer_weight_shapes
        # sys.exit(0)

    def _initialize_layers(self, layer_config):
        """
        Sets the net's <layer> attribute
        to be a list of Layers (classes from layer_type_map)
        """
        layers = []
        input_shape = self.input_shape
        for layer_spec in layer_config:
            # handle the spec format: {'type': {kwargs}}
            # print "layer_spec.keys()[0] ; ", layer_spec.keys()[0]
            layer_class = self.layer_type_map[layer_spec.keys()[0]]  # just one element so [0] is use
            layer_kwargs = layer_spec.values()[0]  # layer arguments
            layer = layer_class(input_shape, **layer_kwargs)  # passing kwards to layers
            input_shape = layer.output.shape  ##set the input shape is the output from layer before
            layers.append(layer)
        self.layers = layers

        for layer in layers:
            log.debug("type : %s | output.shape : %s | weights.shape : %s | biases.shape : %s ",
                      type(layer), layer.output.shape,
                    layer.weights.shape if not isinstance(layer, PoolingLayer) else "None",
                    layer.biases.shape if not isinstance(layer, PoolingLayer) else "None")


        # sys.exit(0)

    def _get_layer_transition(self, inner_ix, outer_ix):


        inner, outer = self.layers[inner_ix], self.layers[outer_ix]
        # print "-----------------------"
        # print "outer_ix : ", outer_ix," outer : ",outer
        # print "inner_ix : ", inner_ix," inner : ",inner


        # either input to FC or pool to FC -> going from 3d matrix to 1d
        # if ((inner_ix < 0 or isinstance(inner, PoolingLayer)) and isinstance(outer, FullyConnectedLayer)):
        if (isinstance(inner, PoolingLayer) and isinstance(outer, FullyConnectedLayer)):
            return '3d_to_1d'

        # going from 3d to 3d matrix -> either input to conv or conv to conv
        # if ((inner_ix < 0 or isinstance(inner, ConvLayer)) and isinstance(outer, ConvLayer)):
        if (isinstance(inner, ConvLayer) and isinstance(outer, ConvLayer)):
            return 'to_conv'

        if (isinstance(inner, FullyConnectedLayer) and (isinstance(outer, ClassifyLayer) or isinstance(outer, FullyConnectedLayer))):
            return '1d_to_1d'

        if (isinstance(inner, ConvLayer) and isinstance(outer, PoolingLayer)):
            return 'conv_to_pool'

        if (isinstance(inner, PoolingLayer) and isinstance(outer, ConvLayer)):
            return 'pool_to_conv'

        raise NotImplementedError

    def feedforward(self, image):
        prev_activation = image


        # forwardpass
        for layer in self.layers:

            input_to_feed = prev_activation

            if isinstance(layer, FullyConnectedLayer):
                # z values are huge, while the fc_output is tiny! large negative vals get penalized to 0!
                layer.feedforward(input_to_feed)
                # print "FullyConnectedLayer : ", layer.output

            elif isinstance(layer, ConvLayer):
                layer.convolve(input_to_feed)
                # for i in range(layer.output.shape[0]):
                #     plt.imsave('images/cat_conv%d.png'%i, layer.output[i])
                # for i in range(layer.weights.shape[0]):
                #     plt.imsave('images/filter_conv%s.png'%i, layer.weights[i].reshape((5,5)))

            elif isinstance(layer, PoolingLayer):
                layer.pool(input_to_feed)
                # for i in range(layer.output.shape[0]):
                #     plt.imsave('images/pool_pic%s.png'%i, layer.output[i])

            elif isinstance(layer, ClassifyLayer):
                layer.classify(input_to_feed)
                # print "Classify : ", layer.output

            else:
                raise NotImplementedError

            prev_activation = layer.output

        # for layer in self.layers:
        #     print "output : ",layer.output.shape,", type(layer) : ",type(layer)
        #
        # sys.exit(0)

        final_activation = prev_activation
        return final_activation

    def backprop(self, image, label):
        nabla_w = [np.zeros(s) for s in self.layer_weight_shapes]  # create nabla_weight for every layer with same shape
        nabla_b = [np.zeros(s) for s in self.layer_biases_shapes]  # create nabla_biases for every layer with same shape

        # print "self.layer_weight_shapes : ", self.layer_weight_shapes
        # print "self.layer_biases_shapes : ", self.layer_biases_shapes

        # print "layer.shape : ", self.layer_weight_shapes
        #
        # for w in nabla_w:
        #     print "nabla_w.shape : ", w.shape
        # for b in nabla_b:
        #     print "nabla_b.shape : ", b.shape
        #
        # sys.exit(0)

        # print "layer weight shapes : ", self.layer_weight_shapes
        # print "layer biases shapes : ", self.layer_biases_shapes

        # set first params on the final layer
        final_output = self.layers[-1].output

        delta = loss_prime(final_output, label) * activation_prime(self.layers[-1].z_values)  # Error * activation_prime(z values layer before)
        last_weights = None
        final = True

        num_layers = len(self.layers)
        # import ipdb;ipdb.set_trace()

        nabla_idx = len(nabla_w) - 1

        for l in range(num_layers - 1, -1, -1):
            # the "outer" layer is closer to classification
            # the "inner" layer is closer to input
            inner_layer_ix = l - 1
            if (l - 1) < 0:
                inner_layer_ix = 0

            outer_layer_ix = l

            layer = self.layers[outer_layer_ix]
            prev_output = self.layers[inner_layer_ix].output if inner_layer_ix >= 0 else image

            layer_next = None
            if (outer_layer_ix < num_layers - 1):
                layer_next = self.layers[outer_layer_ix + 1]

            transition = self._get_layer_transition(
                inner_layer_ix, outer_layer_ix
            )

            log.debug("-----------------")
            log.debug("transition : "+transition)

            # inputfc = poolfc
            # fc to fc = fc to final
            # conv to conv -> input to conv
            # conv to pool -> unique

            if transition == '1d_to_1d':  # final to fc, fc to fc
                db, dw, delta = backprop_1d_to_1d(
                    delta=delta,
                    weights=last_weights,
                    output=prev_output,
                    z_vals=layer.z_values,
                    final=final)
                final = False

            elif transition == '3d_to_1d':
                if l == 0:
                    prev_output = image
                # calc delta on the first final layer
                db, dw, delta = backprop_3d_to_1d(
                    delta=delta,
                    weights=last_weights,  # shape (10,100) this is the weights from the next layer
                    output=prev_output,  # (28,28)
                    z_vals=layer.z_values)  # (100,1)
                # layer.weights = layer.weights.reshape((layer.num_output, layer.depth, layer.height_in, layer.width_in))

            # pool to conv layer
            elif transition == 'conv_to_pool':
                # no update for dw,db => only backprops the error
                delta = backprop_conv_to_pool(
                    delta=delta,
                    weights=last_weights,
                    input_from_conv=prev_output,
                    max_indices=layer.max_indices,
                    poolsize=layer.poolsize,
                    pool_output=layer.output,
                    from_conv=True if isinstance(layer_next, ConvLayer) else False)

            elif transition == 'pool_to_conv':
                # prev_output = image
                # next_weights = layer.weights
                weights_shape = layer.weights.shape
                db, dw = backprop_pool_to_conv(
                    delta=delta,
                    weights_shape=weights_shape,
                    stride=layer.stride,
                    output=prev_output,
                    prev_z_vals=layer.z_values)

            # conv to conv layer
            elif transition == 'to_conv':
                # weights passed in are the ones between conv to conv
                # update the weights and biases
                # prev_output = image
                # last_weights = layer.weights
                weights_shape = layer.weights.shape
                db, dw = backprop_to_conv(
                    delta=delta,
                    weights_shape=layer.weights.shape,
                    stride=layer.stride,
                    output=image,
                    prev_z_vals=layer.z_values)

            else:
                pass

            if transition != 'conv_to_pool':
                # print 'nablasb, db,nabldw, dw, DELTA', nabla_b[inner_layer_ix].shape, db.shape, nabla_w[inner_layer_ix].shape, dw.shape, last_delta.shape
                # print 'outer_layer_ix : ',outer_layer_ix
                # print 'inner_layer_ix : ',inner_layer_ix
                # print 'nabla_idx : ',nabla_idx
                # print 'nabla_w[nabla_idx] : ', nabla_w[nabla_idx].shape
                # print 'dw : ', dw.shape
                # print 'nabla_b[nabla_idx] : ', nabla_b[nabla_idx].shape
                # print 'db : ', db.shape
                nabla_b[nabla_idx], nabla_w[nabla_idx] = db, dw
                last_weights = layer.weights
                nabla_idx -= 1

        # sys.exit(0)

        return self.layers[-1].output, nabla_b, nabla_w

    def gradient_descent(self, training_data, batch_size, eta, num_epochs, num_output, lmbda=None, test_data=None):
        training_size = len(training_data)

        if test_data:
            n_test = len(test_data)

        mean_error = []
        correct_res = []

        log.info('Gradient Descent')
        log.info('batch_size : %s', batch_size)
        log.info('num_epochs : %s', num_epochs)
        log.info('eta : %s', eta)

        for epoch in xrange(num_epochs):
            log.info("Starting epochs : %d", epoch)
            start = time.time()
            random.shuffle(training_data)  # randomize training dataset
            batches = [training_data[k:k + batch_size] for k in xrange(0, training_size, batch_size)]
            losses = 0

            batch_index = 0

            for batch in batches:
                # print '---batch : {}', batch
                log.info( '------- %d', batch_index)
                batch_index += 1
                loss = self.update_mini_batch(batch, eta)
                losses += loss
                log.info( "losses : %s", losses)
            mean_error.append(round(losses / batch_size, 2))
            log.info( "mean error : %s", mean_error)

            if test_data:
                log.info( "################## VALIDATE #################")
                log.info( "Epoch {0} complete %s", format(epoch))
                # res = self.validate(test_data)
                if num_output >= 2:
                    res = self.validate_multiclass(test_data)
                else:
                    res = self.validate(test_data)
                correct_res.append(res)

                # print "res: ", res
                # print "n_tes: ", n_test
                # accuracy = float(res) / float(n_test)
                # print "Accuracy: %.2f" % accuracy
                # time
                timer = time.time() - start
                log.info("Estimated time: %s", timer)
            else:
                log.info("NO TEST DATA")

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(correct_res)
        # plt.show()

    # bisa di paralelisasi
    def update_mini_batch(self, batch, eta):
        nabla_w = [np.zeros(s) for s in self.layer_weight_shapes]
        nabla_b = [np.zeros(s) for s in self.layer_biases_shapes]

        batch_size = len(batch)

        for image, label in batch:
            # image = image.reshape((CHANNEL,HEIGHT,WIDTH))
            image = image.reshape((self.input_shape[0], self.input_shape[1], self.input_shape[2]))

            # print "image.shape : ", image.shape
            # print "label.shape : ", label.shape

            _ = self.feedforward(image)
            final_res, delta_b, delta_w = self.backprop(image, label)
            # sys.exit(0)

            # print 'final_res.shape : ', final_res.shape
            # print 'final_res : ', final_res
            # print 'delta_b.shape : ', delta_b[0].shape
            # print 'len(delta_b) : ', len(delta_b)
            # print 'delta_w.shape : ', delta_w[0].shape
            # print 'len(delta_w) : ', len(delta_w)

            nabla_b = [nb + db for nb, db in zip(nabla_b, delta_b)]  # tambah nilai errornya dengan yang baru
            nabla_w = [nw + dw for nw, dw in zip(nabla_w, delta_w)]  # tambah nilai errornya dengan yang baru

        ################## print LOSS ############
        error = loss(label, final_res)
        log.info("error : %s", error)

        num = 0
        weight_index = []
        for layer in self.layers:
            if not isinstance(layer, PoolingLayer):
                weight_index.append(num)
            num += 1

        for ix, (layer_nabla_w, layer_nabla_b) in enumerate(zip(nabla_w, nabla_b)):
            layer = self.layers[weight_index[ix]]
            # print "type(layer_nabla_w) : ",layer_nabla_w
            # print "type(layer_nabla_b) : ",layer_nabla_b
            # print "type(layer_nabla_b) : ",layer_nabla_b
            layer.weights -= eta * layer_nabla_w / batch_size
            layer.biases -= eta * layer_nabla_b / batch_size
        return error

    def validate(self, data):
        # data = [(im.reshape((CHANNEL,HEIGHT,WIDTH)),y) for im,y in data]
        data = [(im.reshape((self.input_shape[0], self.input_shape[1], self.input_shape[2])), y) for im, y in data]
        # print "data : ",data

        # test_results = [(np.argmax(self.feedforward(x)),y) for x, y in data] #argmax return index of max value

        test_results = list()
        for d in data:
            result = self.feedforward(d[0])
            predicted = np.where(result > 0.5, 1, 0)
            actual = d[1]

            log.info("result : %s | predicted : %s | actual : %s", result, predicted, actual)


            test_results.append((predicted[0][0], actual[0][0]))

        # print "test_results : ", test_results
        # print "len(test_results) : ", len(test_results)
        # output_n = len(test_results) if len(test_results) > 1 else 2

        confusion_matrix = np.zeros([2, 2])
        for test_result in test_results:
            # print "test_results[0] : ", test_results[0]
            # print "test_results[1] : ", test_results[1]
            confusion_matrix[test_result[0]][test_result[1]] += 1
        # print confusion_matrix

        n_test = len(data)

        accuracy = float(np.trace(confusion_matrix)) / n_test

        precision = np.diag(confusion_matrix) / np.sum(confusion_matrix, 0)
        precision = [((p, 0)[math.isnan(p)]) for p in precision]
        recall = np.diag(confusion_matrix) / np.sum(confusion_matrix, 1)
        recall = [((r, 0)[math.isnan(r)]) for r in recall]

        log.info("Confusion Matrix : ")
        log.info(confusion_matrix)
        log.info("Accuracy : %s", accuracy)

        # for multiclass
        log.info("Average Precision : %s", np.average(precision))
        log.info("Average Recall : %s", np.average(recall))

        # for two class
        # log.info( "Precision : ", precision[1])
        # log.info( "Recall : ", recall[1])

        # print type(test_results)
        # return sum(int(x == y) for x, y in test_results)
        return accuracy

    def validate_multiclass(self, data, num_output):
        # data = [(im.reshape((CHANNEL,HEIGHT,WIDTH)),y) for im,y in data]
        data = [(im.reshape((self.input_shape[0], self.input_shape[1], self.input_shape[2])), y) for im, y in data]
        # print "data : ",data

        # test_results = [(np.argmax(self.feedforward(x)),y) for x, y in data] #argmax return index of max value

        test_results = list()
        for d in data:
            result = self.feedforward(d[0])
            # predicted = np.where(result > 0.5, 1, 0)
            actual = d[1]

            # log.info("result : %s | predicted : %s | actual : %s", result, predicted, actual)


            test_results.append((np.argmax(result)+1, actual))

        # print "test_results : ", test_results
        # print "len(test_results) : ", len(test_results)
        # output_n = len(test_results) if len(test_results) > 1 else 2

        confusion_matrix = np.zeros([num_output, num_output])
        for test_result in test_results:
            # print "test_results[0] : ", test_results[0]
            # print "test_results[1] : ", test_results[1]
            confusion_matrix[test_result[0]][test_result[1]] += 1
        # print confusion_matrix

        n_test = len(data)

        accuracy = float(np.trace(confusion_matrix)) / n_test

        precision = np.diag(confusion_matrix) / np.sum(confusion_matrix, 0)
        precision = [((p, 0)[math.isnan(p)]) for p in precision]
        recall = np.diag(confusion_matrix) / np.sum(confusion_matrix, 1)
        recall = [((r, 0)[math.isnan(r)]) for r in recall]

        log.info("Confusion Matrix : ")
        log.info(confusion_matrix)
        log.info("Accuracy : %s", accuracy)

        # for multiclass
        log.info("Average Precision : %s", np.average(precision))
        log.info("Average Recall : %s", np.average(recall))

        # for two class
        # log.info( "Precision : ", precision[1])
        # log.info( "Recall : ", recall[1])

        # print type(test_results)
        # return sum(int(x == y) for x, y in test_results)
        return accuracy

