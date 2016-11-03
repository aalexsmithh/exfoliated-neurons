import sklearn
from sklearn.datasets import make_moons
import numpy as np
import matplotlib.pyplot as plt

#inspired by Andrew Ng's course and examples on writing NN's from scratch, as well as 
#the tutorial at http://www.wildml.com/2015/09/implementing-a-neural-network-from-scratch/


class HandNN:

	def __init__(self, nn_input_dim = 3600, nn_output_dim = 19, nn_hdim=200, epsilon = 0.01, reg_lambda = 0.01, num_examples = 100000):
		self.nn_input_dim = nn_input_dim
		self.nn_output_dim = nn_output_dim
		self.epsilon = epsilon
		self.reg_lambda = reg_lambda
		self.num_examples = num_examples
		self.nn_hdim = nn_hdim

	#Error function to report back error. This cost function uses tanh because its simple to differentiate manually
	def tanh_error(self):
	    W1, b1, W2, b2 = self.model['W1'], self.model['b1'], self.model['W2'], self.model['b2']
	    # Feed forward
	    z1 = self.X.dot(W1) + b1
	    a1 = np.tanh(z1)
	    z2 = a1.dot(W2) + b2
	    exp_scores = np.exp(z2)
	    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
	    corect_logprobs = -np.log(probs[range(self.num_examples), self.y])
	    data_loss = np.sum(corect_logprobs)
	    data_loss += self.reg_lambda/2 * (np.sum(np.square(W1)) + np.sum(np.square(W2)))
	    return 1./self.num_examples * data_loss

	# Same as above, but only feed forward because we don't care about cost when predicting
	def predict(self, x):
	    W1, b1, W2, b2 = self.model['W1'], self.model['b1'], self.model['W2'], self.model['b2']
	    z1 = x.dot(W1) + b1
	    a1 = np.tanh(z1)
	    z2 = a1.dot(W2) + b2
	    exp_scores = np.exp(z2)
	    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
	    return np.argmax(probs, axis=1)

	def fit(self,X,y,max_iters=20000, print_loss=False):
        self.X = X
	    self.y = y
	    # This is the random initialization of parameters
	    np.random.seed(0)
	    W1 = np.random.randn(self.nn_input_dim, self.nn_hdim) / np.sqrt(self.nn_input_dim)
	    b1 = np.zeros((1, self.nn_hdim))
	    W2 = np.random.randn(self.nn_hdim, self.nn_output_dim) / np.sqrt(self.nn_hdim)
	    b2 = np.zeros((1, self.nn_output_dim))
	    self.model = {}
	     
	    # Now we perform gradient descent iterations
	    for i in xrange(0, max_iters):
	 
	        # Feed forward just like in previous methods
	        z1 = X.dot(W1) + b1
	        a1 = np.tanh(z1)
	        z2 = a1.dot(W2) + b2
	        exp_scores = np.exp(z2)
	        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
	 
	        # Backpropagation
	        delta3 = probs
	        print probs.shape, self.num_examples, y.shape
	        delta3[range(self.num_examples), y] -= 1
	        dW2 = (a1.T).dot(delta3)
	        db2 = np.sum(delta3, axis=0, keepdims=True)
	        delta2 = delta3.dot(W2.T) * (1 - np.power(a1, 2))
	        dW1 = np.dot(X.T, delta2)
	        db1 = np.sum(delta2, axis=0)
	        # Add regularization terms
	        dW2 += self.reg_lambda * W2
	        dW1 += self.reg_lambda * W1
	        # Now do updates on parameters
	        W1 += -self.epsilon * dW1
	        b1 += -self.epsilon * db1
	        W2 += -self.epsilon * dW2
	        b2 += -self.epsilon * db2 
	        self.model = { 'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}	         
	        # Print loss if we want (COMPUTATIONALLY EXPENSIVE)
	        if print_loss and i % 1000 == 0:
	          print "Loss after iteration %i: %f" %(i, self.tanh_error(self.model))
	    return self.model
