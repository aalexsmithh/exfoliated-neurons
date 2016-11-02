import numpy as np
import csv, time
import tensorflow as tf

def main():
	BRK_PT = 8000
	END_PT = 10000
	data = np.fromfile('train_x.bin', dtype='uint8')
	data = data.reshape((100000,3600))

	hidden = [3600,1800,900,450,225]#,112,56,28,14,7,3]
	results = []

	f = open('tf_train.log','w')
	f.close()

	for i in reversed(hidden):
		a = time.time()
		score = tf_run(data[0:BRK_PT],data[BRK_PT:END_PT],i)
		elap = time.time() - a
		s = str(i) + " hidden layers gives " + str(score) + " accuracy in " + str(elap/float(60)) + " mins"
		print s
		f = open('tf_train.log','a')
		f.write(s)
		f.write("\n")
		f.close()

	

# Building the encoder
def encoder(x,weights,biases):
	# Encoder Hidden layer with sigmoid activation #1
	layer = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']), biases['encoder_b1']))
	return layer


# Building the decoder
def decoder(x,weights,biases):
	# Encoder Hidden layer with sigmoid activation #1
	layer = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']), biases['decoder_b1']))
	return layer

def score(preds,X,Y):
	score = 0
	tot = 0
	for i,ex in enumerate(X):
		labels = preds[i]
		for j,label in enumerate(labels):
			if label >= Y[i][j]-10 and label <= Y[i][j]+10:
				score += 1
				tot += 1
			else:
				tot += 1
	print score, tot
	return float(score)/float(tot)
		
def tf_run(data_train,data_test,n_hid,verbose=False):
	# Parameters
	learning_rate = 0.01
	max_epochs = 1000
	train_cost_threshold = 1e-3
	batch_size = 2500
	display_step = 1
	examples_to_show = 10

	# Network Parameters
	n_hidden_1 = n_hid # 1st layer num features
	n_input = 3600 # MNIST data input (img shape: 28*28)

	# tf Graph input (only pictures)
	X = tf.placeholder("float", [None, n_input])

	weights = {
		'encoder_h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
		'decoder_h1': tf.Variable(tf.random_normal([n_hidden_1, n_input])),
	}
	biases = {
		'encoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
		'decoder_b1': tf.Variable(tf.random_normal([n_input])),
	}


	# Construct model
	encoder_op = encoder(X,weights,biases)
	decoder_op = decoder(encoder_op,weights,biases)

	# Prediction
	y_pred = decoder_op
	# Targets (Labels) are the input data.
	y_true = X

	# Define loss and optimizer, minimize the squared error
	cost = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
	optimizer = tf.train.RMSPropOptimizer(learning_rate,decay=0.9).minimize(cost)

	# Initializing the variables
	init = tf.initialize_all_variables()

	# Launch the graph
	with tf.Session() as sess:
		sess.run(init)
		total_batch = len(data_train)/batch_size
		# Training cycle
		costs = []
		for epoch in range(max_epochs):
			# Loop over all batches
			for i in range(total_batch):
				batch_xs = data_train[(i*batch_size):((i+1)*batch_size)]
				# Run optimization op (backprop) and cost op (to get loss value)
				_, c = sess.run([optimizer, cost], feed_dict={X: batch_xs})
			costs.append(c)
			# Display logs per epoch step
			if epoch % display_step == 0 and verbose:
				print "Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(c)
			if epoch != 0 and epoch != 1:
				if abs(costs[-2] - costs[-1]) < train_cost_threshold:
					if verbose:
						print "Training reached threshold. Breaking..."
					break

		# Applying encode and decode over test set
		encode_decode = sess.run(
			y_pred, feed_dict={X: data_test})
		# Compare original images with their reconstructions
		acc = score(encode_decode,data_test,data_test)
		return acc

if __name__ == '__main__':
	main()

