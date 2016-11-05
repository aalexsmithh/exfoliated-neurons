'''Trains a simple convnet on the MNIST dataset.
Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.
'''

from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
import csv

batch_size = 128
nb_classes = 19
nb_epoch = 12
BRK_PT = 70000
END_PT = 100000

# input image dimensions
img_rows, img_cols = 60, 60
# number of convolutional filters to use
nb_filters = 64
# size of pooling area for max pooling
pool_size = (2, 2)
# convolution kernel size
kernel_size = (9, 9)

def main():

	X = np.fromfile('data/train_x.bin', dtype='uint8')
	X = X.reshape((100000,60,60))
	Y = open_csv("data/train_y.csv")

	X_train = X[0:BRK_PT]
	y_train = Y[0:BRK_PT]

	X_test = X[BRK_PT:END_PT]
	y_test = Y[BRK_PT:END_PT]

	if K.image_dim_ordering() == 'th':
	    X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
	    X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
	    input_shape = (1, img_rows, img_cols)
	else:
	    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
	    X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
	    input_shape = (img_rows, img_cols, 1)

	X_train = X_train.astype('float32')
	X_test = X_test.astype('float32')
	X_train /= 255
	X_test /= 255
	print('X_train shape:', X_train.shape)
	print(X_train.shape[0], 'train samples')
	print(X_test.shape[0], 'test samples')

	# # convert class vectors to binary class matrices
	Y_train = np_utils.to_categorical(y_train, nb_classes)
	Y_test = np_utils.to_categorical(y_test, nb_classes)

	print (Y_train[0])

	model = Sequential()

	model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1],
	                        border_mode='valid',
	                        input_shape=input_shape))
	model.add(Activation('relu'))
	model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1]))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=pool_size))
	model.add(Dropout(0.25))

	model.add(Flatten())
	model.add(Dense(128))
	model.add(Activation('relu'))
	model.add(Dropout(0.5))
	model.add(Dense(nb_classes))
	model.add(Activation('softmax'))

	model.compile(loss='categorical_crossentropy',
	              optimizer='adadelta',
	              metrics=['accuracy'])

	model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
	          verbose=1, validation_data=(X_test, Y_test))
	score = model.evaluate(X_test, Y_test, verbose=0)
	print('Test score:', score[0])
	print('Test accuracy:', score[1])

def open_csv(filename):
	with open(filename, 'rb') as csvfile:
		ret = []
		csv_in = csv.reader(csvfile, delimiter=',', quotechar='"')
		for line in csv_in:
			ret.append(line[1])
		return np.asarray(ret[1:],dtype='uint')


if __name__ == '__main__':
	main()