from MNIST_trained import MNIST_Model
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
step = 4
mnm = MNIST_Model()
x_data = np.fromfile('../train_x.bin', dtype='uint8')
x_data = x_data.reshape((100000,60,60))
def get(image):
	return np.array([image[i:i+28,j:j+28] for i in range(0,32,step) for j in range(0,32,step)])

num_si = 32**2 // step **2
subimages = map(get, x_data)
flatten = lambda l: [i for sl in l for i in sl]
shaper = lambda l: np.array(l).reshape(num_si*len(l), 1, 28, 28)
results = flatten(map(mnm.predict, [shaper(subimages[i*500:(i+1)*500]) for i in xrange(len(subimages)//500)]))
results = np.array(results).reshape(100000,num_si*10)
with open("../x_subimages.csv", "wb") as f:
    writer = csv.writer(f)
    writer.writerows(results)


