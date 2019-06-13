import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
import number_loader
import hanacaraka_loader
import mammogram_loader
from keras import backend as K
import sys

batch_size = 32
num_classes = 2
epochs = 50

# input image dimensions

# the data, split between train and test sets
# (x_train, y_train), (x_test, y_test) = mnist.load_data()
# img_rows, img_cols , (x_train, y_train), (x_test, y_test) = number_loader.load_data()
img_rows, img_cols , (x_train, y_train), (x_test, y_test) = hanacaraka_loader.load_data()
# img_rows, img_cols , (x_train, y_train), (x_test, y_test) = mammogram_loader.load_data("real")


if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
# x_train /= 255
# x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

print "x_train.shape : ",x_train.shape
print "x_test.shape : ",x_test.shape
print "y_train.shape : ",y_train.shape
print "y_test.shape : ",y_test.shape




model = Sequential()
model.add(Conv2D(20, kernel_size=(5, 5), activation='relu', input_shape=input_shape))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Conv2D(50, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(30, activation='relu'))
# model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='sigmoid'))

optimizer = keras.optimizers.SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False)
# optimizer = keras.optimizers.Adadelta()

model.compile(loss=keras.losses.binary_crossentropy, optimizer=optimizer, metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=1)

print('Test loss:', score[0])
print('Test accuracy:', score[1])

y_predict  = model.predict_classes(x_test)
# print "y_predict : ",y_predict
# print "y_test : ",y_test