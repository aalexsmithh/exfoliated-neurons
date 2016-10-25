import sklearn
from sklearn.datasets import make_moons
import numpy as np
import matplotlib.pyplot as plt

#inspired by following the tutorial at http://www.wildml.com/2015/09/implementing-a-neural-network-from-scratch/


class HandNN:

	def __init__(self, nn_input_dim = 3600, nn_output_dim = 19, nn_hdim=200, epsilon = 0.01, reg_lambda = 0.01, num_examples = 100000):
		self.nn_input_dim = nn_input_dim
		self.nn_output_dim = nn_output_dim
		self.epsilon = epsilon
		self.reg_lambda = reg_lambda
		self.num_examples = num_examples
		self.nn_hdim = nn_hdim

	# Helper function to evaluate the total loss on the dataset
	def tanh_error(self):
	    W1, b1, W2, b2 = self.model['W1'], self.model['b1'], self.model['W2'], self.model['b2']
	    # Forward propagation to calculate our predictions
	    z1 = self.X.dot(W1) + b1
	    a1 = np.tanh(z1)
	    z2 = a1.dot(W2) + b2
	    exp_scores = np.exp(z2)
	    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
	    # Calculating the loss
	    corect_logprobs = -np.log(probs[range(self.num_examples), self.y])
	    data_loss = np.sum(corect_logprobs)
	    # Add regulatization term to loss (optional)
	    data_loss += self.reg_lambda/2 * (np.sum(np.square(W1)) + np.sum(np.square(W2)))
	    return 1./self.num_examples * data_loss

	# Helper function to predict an output (0 or 1)
	def predict(self, x):
	    W1, b1, W2, b2 = self.model['W1'], self.model['b1'], self.model['W2'], self.model['b2']
	    # Forward propagation
	    z1 = x.dot(W1) + b1
	    a1 = np.tanh(z1)
	    z2 = a1.dot(W2) + b2
	    exp_scores = np.exp(z2)
	    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
	    return np.argmax(probs, axis=1)

	def fit(self,X,y,max_iters=20000, print_loss=False):
            self.X = X
	    self.y = y
	    # Initialize the parameters to random values. We need to learn these.
	    np.random.seed(0)
	    W1 = np.random.randn(self.nn_input_dim, self.nn_hdim) / np.sqrt(self.nn_input_dim)
	    b1 = np.zeros((1, self.nn_hdim))
	    W2 = np.random.randn(self.nn_hdim, self.nn_output_dim) / np.sqrt(self.nn_hdim)
	    b2 = np.zeros((1, self.nn_output_dim))
	 
	    # This is what we return at the end
	    self.model = {}
	     
	    # Gradient descent. For each batch...
	    for i in xrange(0, max_iters):
	 
	        # Forward propagation
	        z1 = X.dot(W1) + b1
	        a1 = np.tanh(z1)
	        z2 = a1.dot(W2) + b2
	        exp_scores = np.exp(z2)
	        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
	 
	        # Backpropagation
	        delta3 = probs
	        delta3[range(self.num_examples), y] -= 1
	        dW2 = (a1.T).dot(delta3)
	        db2 = np.sum(delta3, axis=0, keepdims=True)
	        delta2 = delta3.dot(W2.T) * (1 - np.power(a1, 2))
	        dW1 = np.dot(X.T, delta2)
	        db1 = np.sum(delta2, axis=0)
	 
	        # Add regularization terms (b1 and b2 don't have regularization terms)
	        dW2 += self.reg_lambda * W2
	        dW1 += self.reg_lambda * W1
	 
	        # Gradient descent parameter update
	        W1 += -self.epsilon * dW1
	        b1 += -self.epsilon * db1
	        W2 += -self.epsilon * dW2
	        b2 += -self.epsilon * db2
	         
	        # Assign new parameters to the self.model
	        self.model = { 'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}
	         
	        # Optionally print the loss.
	        # This is expensive because it uses the whole dataset, so we don't want to do it too often.
	        if print_loss and i % 1000 == 0:
	          print "Loss after iteration %i: %f" %(i, self.tanh_error(self.model))
	    return self.model
