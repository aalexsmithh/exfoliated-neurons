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

class MNIST_Model:
	def __init__(self):
		num_units = 100
		data_size=(None,1,28,28) # Batch size x Img Channels x Height x Width
		output_size=10 # We will run the example in mnist - 10 digits
		input_var = T.tensor4('input')
		target_var = T.ivector('targets')
		net = {}
		#Input layer:
		net['data'] = lasagne.layers.InputLayer(data_size, input_var=input_var)
		#Convolution + Pooling
		net['conv1'] = lasagne.layers.Conv2DLayer(net['data'], num_filters=6, filter_size=5)
		net['pool1'] = lasagne.layers.Pool2DLayer(net['conv1'], pool_size=2)
		net['conv2'] = lasagne.layers.Conv2DLayer(net['pool1'], num_filters=10, filter_size=5)
		net['pool2'] = lasagne.layers.Pool2DLayer(net['conv2'], pool_size=2)
		net['fc1'] = lasagne.layers.DenseLayer(net['pool2'], num_units=num_units)
		net['drop1'] = lasagne.layers.DropoutLayer(net['fc1'],  p=0.5)
		net['out'] = lasagne.layers.DenseLayer(net['drop1'], num_units=output_size, 
		                                       nonlinearity=lasagne.nonlinearities.softmax)
		with open(r"ALL_PARAMS.pickle", "rb") as input_file:
		    ALL_PARAMS = cPickle.load(input_file)
		lasagne.layers.set_all_param_values(net['out'], ALL_PARAMS)
		#Define hyperparameters. These could also be symbolic variables 
		lr = 1e-2 #learning rate
		weight_decay = 1e-5 #L2 regularization
		#Loss function: mean cross-entropy
		prediction = lasagne.layers.get_output(net['out'])
		loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
		loss = loss.mean()
		#Also add weight decay to the cost function
		weightsl2 = lasagne.regularization.regularize_network_params(net['out'], lasagne.regularization.l2)
		loss += weight_decay * weightsl2
		#Get the update rule for Stochastic Gradient Descent with Nesterov Momentum
		params = lasagne.layers.get_all_params(net['out'], trainable=True)
		updates = lasagne.updates.sgd(
		        loss, params, learning_rate=lr)
		self.train_fn = theano.function([input_var, target_var], loss, updates=updates)
		test_prediction = lasagne.layers.get_output(net['out'], deterministic=True)
		test_loss = lasagne.objectives.categorical_crossentropy(test_prediction,
		                                                        target_var)
		test_loss = test_loss.mean()
		test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),
		                  dtype=theano.config.floatX)
		self.val_fn = theano.function([input_var, target_var], [test_loss, test_acc])
		self.get_preds = theano.function([input_var], test_prediction)
		output = lasagne.layers.get_output(net['out'])
		self.get_output = theano.function([input_var], output)

	def predict(self, input_img):
		return self.get_output(input_img.reshape(-1,1,28,28))

