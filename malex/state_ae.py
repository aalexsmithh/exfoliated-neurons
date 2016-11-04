from __future__ import print_function

import numpy as np
import theano
import lasagne

# Used some source code from the following link, which inspired this code:
# https://github.com/goelhardik/autoencoder-lasagne

# Size of encoded representation
N_ENCODED = 7200
# Number of epochs to train the net
NUM_EPOCHS = 100
# Batch Size
BATCH_SIZE = 500
# Input feature size
NUM_FEATURES = 60 * 60

##################################################################
# Function to generate batches of data starting from the index p

def gen_data(data, p, batch_size = BATCH_SIZE, return_target=True):

    x = np.zeros((batch_size,NUM_FEATURES))
    for n in range(batch_size):
        x[n,:] = data[p+n, :]

    return x, x

#################################################################
# Function that builds the network, given the parameters

def build_network():
    print("Building network ...")
       
    # First, we build the network, starting with an input layer
    # Shape is
    # (batch size, NUM_FEATURES)
    
    l_in = lasagne.layers.InputLayer(shape=(BATCH_SIZE, NUM_FEATURES))
    
    # This is the encoding layer, the output of this stage is (batch_size, N_ENCODED)
    encoder_l_out = lasagne.layers.DenseLayer(l_in, num_units=N_ENCODED, W = 
                                              lasagne.init.Normal(), 
                                              nonlinearity=lasagne.nonlinearities.rectify)
    
   
    # This is the decoding layer; outputs (batch_size, NUM_FEATURES)
    decoder_l_out = lasagne.layers.DenseLayer(encoder_l_out, num_units = NUM_FEATURES, 
                                              W = lasagne.init.Normal(), 
                                              nonlinearity = 
                                              lasagne.nonlinearities.sigmoid)
    
    # Theano tensor for the targets
    target_values = theano.tensor.fmatrix('target_output')
    
    # lasagne.layers.get_output produces a variable for the encoded value
    encoded_output = lasagne.layers.get_output(encoder_l_out)
    
    # lasagne.layers.get_output produces a variable for the output of the net
    network_output = lasagne.layers.get_output(decoder_l_out)
    
    # The loss function is calculated as the mean of the squared error between the prediction and target.
    cost = lasagne.objectives.squared_error(network_output,target_values).mean()
    
    # Retrieve all parameters from the network
    all_params = lasagne.layers.get_all_params(decoder_l_out,trainable=True)
    
    # Compute AdaDelta updates for training
    print("Computing updates ...")
    updates = lasagne.updates.adadelta(cost, all_params)
    
    # Theano functions for training and computing cost
    print("Compiling functions ...")
    train = theano.function([l_in.input_var, target_values], cost, updates=updates, allow_input_downcast=True)
    
    # Theano function to get the reconstructed values
    predict = theano.function([l_in.input_var], network_output, allow_input_downcast=True)
    
    # Theano function to get the encoded values
    encode = theano.function([l_in.input_var], encoded_output, allow_input_downcast=True)

    return train, predict, encode