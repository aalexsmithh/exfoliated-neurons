from __future__ import print_function
import numpy as np
import theano
from theano import tensor as T
import lasagne
import csv
import time
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import sys
import os
import gzip
import cPickle
import urllib
import cPickle
from sklearn.utils import shuffle
from MNIST_trained import MNIST_Model

print("Setting up/Compiling....")

# Used some source code from the following link, which inspired this code:
# http://luizgh.github.io/libraries/2015/12/08/getting-started-with-lasagne/

####################################
num_units = 10
layers = 3
conv = False
wid = 64
ht = 10
#Define hyperparameters. These could also be symbolic variables 
lr = 1 #learning rate
weight_decay = 1e-5 #L2 regularization
####################################
net = {}
data_size=(None,1,wid,ht) # Batch size x Img Channels x Height x Width
output_size=19 # We will run the example in mnist - 10 digits
input_var = T.tensor4('input')
target_var = T.ivector('targets')



if conv:
	#Input layer:
	net['data'] = lasagne.layers.InputLayer(data_size, input_var=input_var)
	#Convolution + Pooling
	net['conv1'] = lasagne.layers.Conv2DLayer(net['data'], num_filters=6, filter_size=5)
	net['pool1'] = lasagne.layers.Pool2DLayer(net['conv1'], pool_size=2)

	net['conv2'] = lasagne.layers.Conv2DLayer(net['pool1'], num_filters=10, filter_size=5)
	net['pool2'] = lasagne.layers.Pool2DLayer(net['conv2'], pool_size=2)
else:
	net['pool2'] = lasagne.layers.InputLayer(data_size, input_var=input_var)


#Fully-connected + dropout
if layers == 3:
	net['fc1'] = lasagne.layers.DenseLayer(net['pool2'], num_units=num_units, nonlinearity=lasagne.nonlinearities.rectify)
	net['drop1'] = lasagne.layers.DropoutLayer(net['fc1'],  p=0.2)
	net['fc2'] = lasagne.layers.DenseLayer(net['drop1'], num_units=num_units, nonlinearity=lasagne.nonlinearities.rectify)
	net['drop2'] = lasagne.layers.DropoutLayer(net['fc2'],  p=0.2)
	net['fc3'] = lasagne.layers.DenseLayer(net['drop2'], num_units=num_units, nonlinearity=lasagne.nonlinearities.rectify)
	net['drop3'] = lasagne.layers.DropoutLayer(net['fc3'],  p=0.2)

	#Output layer:
	net['out'] = lasagne.layers.DenseLayer(net['drop3'], num_units=output_size, 
	                                       nonlinearity=lasagne.nonlinearities.softmax)
elif layers == 2:
	net['fc1'] = lasagne.layers.DenseLayer(net['pool2'], num_units=num_units)
	net['drop1'] = lasagne.layers.DropoutLayer(net['fc1'],  p=0.5)
	net['fc2'] = lasagne.layers.DenseLayer(net['drop1'], num_units=num_units)
	net['drop2'] = lasagne.layers.DropoutLayer(net['fc2'],  p=0.5)

	#Output layer:
	net['out'] = lasagne.layers.DenseLayer(net['drop2'], num_units=output_size, 
	                                       nonlinearity=lasagne.nonlinearities.softmax)
else:
	net['fc1'] = lasagne.layers.DenseLayer(net['pool2'], num_units=num_units)
	net['drop1'] = lasagne.layers.DropoutLayer(net['fc1'],  p=0.5)
	net['out'] = lasagne.layers.DenseLayer(net['drop1'], num_units=output_size, 
	                                       nonlinearity=lasagne.nonlinearities.softmax)




#Loss function: mean cross-entropy
prediction = lasagne.layers.get_output(net['out'])
loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
loss = loss.mean()

#Also add weight decay to the cost function
weightsl2 = lasagne.regularization.regularize_network_params(net['out'], lasagne.regularization.l2)
loss += weight_decay * weightsl2

#Get the update rule for Stochastic Gradient Descent with Nesterov Momentum
params = lasagne.layers.get_all_params(net['out'], trainable=True)
updates = lasagne.updates.nesterov_momentum(
            loss,
            params,
            learning_rate=lr,
            momentum=0.9)

train_fn = theano.function([input_var, target_var], loss, updates=updates)

test_prediction = lasagne.layers.get_output(net['out'], deterministic=True)
test_loss = lasagne.objectives.categorical_crossentropy(test_prediction,
                                                        target_var)
test_loss = test_loss.mean()
test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),
                  dtype=theano.config.floatX)

val_fn = theano.function([input_var, target_var], [test_loss, test_acc])
get_preds = theano.function([input_var], test_prediction)


##########################################################################################
if len(sys.argv) != 2:
	print("Loading data...")
	def open_csv(filename):
		with open(filename, 'rb') as csvfile:
			ret = []
			csv_in = csv.reader(csvfile, delimiter=',', quotechar='"')
			for line in csv_in:
				ret.append(line[1])
			return ret[1:]
	x_data = np.genfromtxt('../x_subimages.csv', delimiter=',')
	y_data = open_csv('../train_y.csv')
	x_data, y_data = shuffle(x_data, y_data, random_state=0)
	x_train = x_data[:90000, :]
	x_test = x_data[90000:, :]
	y_train = np.array(y_data[:90000])
	y_test = np.array(y_data[90000:])
	x_train = x_train.reshape(-1, 1, wid, ht)
	x_test = x_test.reshape(-1, 1, wid, ht)
	y_train = y_train.astype(np.int32)
	y_test = y_test.astype(np.int32)

	epochs = 500
	batch_size= 1000

	n_examples = x_train.shape[0]
	n_batches = n_examples / batch_size

	print("Beginning training...")

	start_time = time.time()

	cost_history = []
	for epoch in xrange(epochs):
	    st = time.time()
	    batch_cost_history = []
	    for batch in xrange(n_batches):
	        x_batch = x_train[batch*batch_size: (batch+1) * batch_size]
	        y_batch = y_train[batch*batch_size: (batch+1) * batch_size]
	        this_cost = train_fn(x_batch, y_batch)
	        batch_cost_history.append(this_cost)
	    epoch_cost = np.mean(batch_cost_history)
	    cost_history.append(epoch_cost)
	    en = time.time()
	    print('Epoch %d/%d, train error: %f. Elapsed time: %.2f seconds' % (epoch+1, epochs, epoch_cost, en-st))
	    loss, acc = val_fn(x_test, y_test)
	    _, tacc = val_fn(x_train, y_train)
	    test_error = 1 - acc
	    print('Train accuracy: %f' % tacc)
	    print('Test accuracy: %f' % acc)
	    # if len(cost_history) > 2:
	    # 	if abs((cost_history[-1] - cost_history[-2])/cost_history[-2]) < 0.01:
	    # 		break
	    
	end_time = time.time()
	print('Training completed in %.2f seconds.' % (end_time - start_time))
	ALL_PARAMS = lasagne.layers.get_all_param_values(net['out'])
	with open(r"DNN_Params.pickle", "wb") as output_file:
		cPickle.dump(ALL_PARAMS, output_file)

else:
	def open_csv(filename):
		with open(filename, 'rb') as csvfile:
			ret = []
			csv_in = csv.reader(csvfile, delimiter=',', quotechar='"')
			for line in csv_in:
				ret.append(line[1])
			return ret[1:]
	x_data = np.genfromtxt('../x_subimages.csv', delimiter=',')
	y_data = open_csv('../train_y.csv')
	x_train = x_data[:90000, :]
	x_test = x_data[90000:, :]
	y_train = np.array(y_data[:90000])
	y_test = np.array(y_data[90000:])
	x_train = x_train.reshape(-1, 1, wid, ht)
	x_test = x_test.reshape(-1, 1, wid, ht)
	y_train = y_train.astype(np.int32)
	y_test = y_test.astype(np.int32)
	with open(r"DNN_Params.pickle", "rb") as input_file:
		ALL_PARAMS = cPickle.load(input_file)
		lasagne.layers.set_all_param_values(net['out'], ALL_PARAMS)
	output = lasagne.layers.get_output(net['out'])
	get_output = theano.function([input_var], output)
	test_accuracy = get_acc(x_test,y_test)
	print('Test accuracy: %f' % test_acc)
	
