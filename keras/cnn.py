from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility
# from keras.datasets import mnist
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.models import load_model
from keras import backend as K
import csv
import sys

batch_size = 150
nb_classes = 19
nb_epoch = 100

earlyStopping=keras.callbacks.EarlyStopping(monitor='val_loss', patience=8, verbose=0, mode='auto')
saver = keras.callbacks.ModelCheckpoint('./128_64_final_output2', monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto')
logger = keras.callbacks.CSVLogger('J_over_time2.csv', separator=',', append=False)
# input image dimensions
img_rows, img_cols = 60, 60
# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
pool_size = (2, 2)
# convolution kernel size
kernel_size = (5, 5)

# the data, shuffled and split between train and test sets
def open_csv(filename):
  with open(filename, 'rb') as csvfile:
    ret = []
    csv_in = csv.reader(csvfile, delimiter=',', quotechar='"')
    for line in csv_in:
      ret.append(line[1])
    return ret[:]

# X_data = np.fromfile('../train_x.bin', dtype='uint8')
# y_data = open_csv('../train_y.csv')
# print(y_data[0])
# print(len(y_data))
# X_data = X_data.reshape((100000,60,60))
# X_train = X_data[:90000]
# X_test = X_data[90000:]
# y_train = np.array(y_data[:90000])
# y_test = np.array(y_data[90000:])
# y_train = y_train.astype(np.int32)
# y_test = y_test.astype(np.int32)

X_train = np.fromfile('../train_x.bin', dtype='uint8')
X_test = np.fromfile('../test_x.bin', dtype='uint8')
y_train = np.array(open_csv('../train_y.csv'))
X_train = X_train.reshape((100000,60,60))
X_test = X_test.reshape((len(X_test)/3600,60,60))
y_train = y_train.astype(np.int32)

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
X_train /= 255.0
X_test /= 255.0
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
# Y_test = np_utils.to_categorical(y_test, nb_classes)

if len(sys.argv) != 2:
  model = Sequential()

  model.add(Convolution2D(30, 5, 5,
                          border_mode='valid',
                          input_shape=input_shape))
  model.add(Activation('relu'))
  model.add(MaxPooling2D(pool_size=pool_size))
  model.add(Convolution2D(15, 3, 3))
  model.add(Activation('relu'))
  model.add(MaxPooling2D(pool_size=pool_size))
  model.add(Dropout(0.2))

  model.add(Flatten())
  model.add(Dense(128))
  model.add(Activation('relu'))
  model.add(Dense(32))
  model.add(Activation('relu'))
  model.add(Dense(nb_classes))
  model.add(Activation('softmax'))

  model.compile(loss='categorical_crossentropy',
                optimizer='adadelta',
                metrics=['accuracy'])

else:
  model = load_model(sys.argv[1])

model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
          verbose=1, callbacks=[logger])#, callbacks=[earlyStopping, saver], validation_data=(X_test, Y_test))
# score = model.evaluate(X_test, Y_test, verbose=0)
# print('Test score:', score[0])
# print('Test accuracy:', score[1])

model.save(str(nb_epoch) + 'outputter2.model')

classes = model.predict_classes(X_test, batch_size=32)
output = [['id', 'category']]
for i in range(len(classes)):
  output.append([i, classes[i]])
with open("mal_output_128_64_final_output2.csv", "wb") as f:
    writer = csv.writer(f)
    writer.writerows(output)