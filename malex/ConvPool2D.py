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

print("Setting up/Compiling....")

# Used some source code from the following link, which inspired this code:
# http://luizgh.github.io/libraries/2015/12/08/getting-started-with-lasagne/

####################################
num_units = 128
layers = 1
conv = True
original_mnist = False
####################################
net = {}
if original_mnist:
	data_size=(None,1,28,28) # Batch size x Img Channels x Height x Width
	output_size=10 # We will run the example in mnist - 10 digits
else:
	data_size=(None,1,60,60) # Batch size x Img Channels x Height x Width
	output_size=19 # We will run the example in mnist - 10 digits
input_var = T.tensor4('input')
target_var = T.ivector('targets')



if conv:
	#Input layer:
	net['data'] = lasagne.layers.InputLayer(data_size, input_var=input_var)
	#Convolution + Pooling
	net['conv1'] = lasagne.layers.Conv2DLayer(net['data'], num_filters=32, filter_size=3)
	net['pool1'] = lasagne.layers.Pool2DLayer(net['conv1'], pool_size=2)

	net['conv2'] = lasagne.layers.Conv2DLayer(net['pool1'], num_filters=32, filter_size=3)
	net['pool2'] = lasagne.layers.Pool2DLayer(net['conv2'], pool_size=2)
else:
	net['pool2'] = lasagne.layers.InputLayer(data_size, input_var=input_var)


#Fully-connected + dropout
if layers == 3:
	net['fc1'] = lasagne.layers.DenseLayer(net['pool2'], num_units=num_units)
	net['drop1'] = lasagne.layers.DropoutLayer(net['fc1'],  p=0.5)
	net['fc2'] = lasagne.layers.DenseLayer(net['drop1'], num_units=num_units)
	net['drop2'] = lasagne.layers.DropoutLayer(net['fc2'],  p=0.5)
	net['fc3'] = lasagne.layers.DenseLayer(net['drop2'], num_units=num_units)
	net['drop3'] = lasagne.layers.DropoutLayer(net['fc3'],  p=0.5)

	#Output layer:
	net['out'] = lasagne.layers.DenseLayer(net['drop3'], num_units=output_size, 
	                                       nonlinearity=lasagne.nonlinearities.softmax)
elif layers == 2:
	net['fc1'] = lasagne.layers.DenseLayer(net['pool2'], num_units=num_units)
	net['drop1'] = lasagne.layers.DropoutLayer(net['fc1'],  p=0.5)
	net['fc2'] = lasagne.layers.DenseLayer(net['drop1'], num_units=num_units)

	#Output layer:
	net['out'] = lasagne.layers.DenseLayer(net['fc2'], num_units=output_size, 
	                                       nonlinearity=lasagne.nonlinearities.softmax)
else:
	net['fc1'] = lasagne.layers.DenseLayer(net['pool2'], num_units=num_units)
	net['drop1'] = lasagne.layers.DropoutLayer(net['fc1'],  p=0.5)
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
print("Loading data...")

def open_csv(filename):
	with open(filename, 'rb') as csvfile:
		ret = []
		csv_in = csv.reader(csvfile, delimiter=',', quotechar='"')
		for line in csv_in:
			ret.append(line[1])
		return ret[1:]

if original_mnist:
	dataset_file = 'mnist.pkl.gz'
	#Download dataset if not yet done:
	if not os.path.isfile(dataset_file):
	    urllib.urlretrieve('http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz', dataset_file)
	#Load the dataset
	f = gzip.open(dataset_file, 'rb')
	train_set, valid_set, test_set = cPickle.load(f)
	f.close()
	#Convert the dataset to the shape we want
	x_train, y_train = train_set
	x_test, y_test = test_set
	x_train = x_train.reshape(-1, 1, 28, 28)
	x_test = x_test.reshape(-1, 1, 28, 28)
	y_train = y_train.astype(np.int32)
	y_test = y_test.astype(np.int32)

else:
	x_data = np.fromfile('../train_x.bin', dtype='uint8')
	y_data = open_csv('../train_y.csv')
	x_data = x_data.reshape((100000,3600)) / 255
	x_train = x_data[:90000, :]
	x_test = x_data[90000:, :]
	y_train = np.array(y_data[:90000])
	y_test = np.array(y_data[90000:])
	x_train = x_train.reshape(-1, 1, 60, 60)
	x_test = x_test.reshape(-1, 1, 60, 60)
	y_train = y_train.astype(np.int32)
	y_test = y_test.astype(np.int32)

epochs = 50
batch_size=1000

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
        
        this_cost = train_fn(x_batch, y_batch) # This is where the model gets updated
        
        batch_cost_history.append(this_cost)
    epoch_cost = np.mean(batch_cost_history)
    cost_history.append(epoch_cost)
    en = time.time()
    print('Epoch %d/%d, train error: %f. Elapsed time: %.2f seconds' % (epoch+1, epochs, epoch_cost, en-st))
    loss, acc = val_fn(x_test, y_test)
    test_error = 1 - acc
    # _, tacc = val_fn(x_train, y_train)
    # train_error = 1-tacc
    # print('Train error: %f' % train_error)
    print('Test error: %f' % test_error)
    if len(cost_history) > 2:
    	if abs((cost_history[-1] - cost_history[-2])/cost_history[-2]) < 0.01:
    		break
    
end_time = time.time()
print('Training completed in %.2f seconds.' % (end_time - start_time))
ALL_PARAMS = lasagne.layers.get_all_param_values(net['out'])
with open(r"ALL_PARAMS.pickle", "wb") as output_file:
	cPickle.dump(ALL_PARAMS, output_file)


###########################################################################################

print("Beginning testing...")

start_time = time.time()

loss, acc = val_fn(x_test, y_test)
test_error = 1 - acc
print('Test error: %f' % test_error)

end_time = time.time()
print('Classifying %d images completed in %.2f seconds.' % (x_test.shape[0], end_time - start_time))

plt.plot(cost_history)
plt.savefig(str(num_units) + "J_over_time.png")

def PlotPredictions(test_samples, predictedLabels):
    plt.figure(figsize=(3,2))
    f, axarr = plt.subplots(2,5)
    for i in range(2):
        for j in range(5):
            index = i*5+j
            axarr[i,j].imshow(test_samples[index].reshape(60,60), cmap='Greys', interpolation='nearest')
            axarr[i,j].axis('off')
            axarr[i,j].set_title('%d' % predictedLabels[index])

predictedLabels = get_preds(x_test).argmax(axis=1)

indices = np.random.choice(y_test.shape[0], replace=False, size=10)

PlotPredictions( x_test[indices], predictedLabels[indices])

plt.savefig(str(num_units) + "randompreds.png")

incorrectPreds = np.nonzero(predictedLabels != y_test)[0]
someIncorrectPreds = np.random.choice(incorrectPreds, replace=False, size=10)

PlotPredictions(x_test[someIncorrectPreds], predictedLabels[someIncorrectPreds])

plt.savefig(str(num_units) + "errors.png")


# weights = net['conv1'].W.get_value()

# plt.figure()
# f, axarr = plt.subplots(3,2)
# for i in range(3):
#     for j in range(2):
#         index = i*2+j
#         axarr[i,j].imshow(weights[index].reshape(5,5), cmap='Greys_r', interpolation='nearest')
#         axarr[i,j].axis('off')

# plt.savefig(str(num_units) + "weights.png")
















