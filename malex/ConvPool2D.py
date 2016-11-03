from __future__ import print_function

import numpy as np
import theano
from theano import tensor as T
import lasagne

print "Setting up/Compiling...."

data_size=(None,1,60,60) # Batch size x Img Channels x Height x Width
output_size=19 # We will run the example in mnist - 10 digits

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


#Fully-connected + dropout
net['fc1'] = lasagne.layers.DenseLayer(net['pool2'], num_units=100)
net['drop1'] = lasagne.layers.DropoutLayer(net['fc1'],  p=0.5)

#Output layer:
net['out'] = lasagne.layers.DenseLayer(net['drop1'], num_units=output_size, 
                                       nonlinearity=lasagne.nonlinearities.softmax)

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
data = np.fromfile('../train_x.bin', dtype='uint8')

n_examples = x_train.shape[0]














