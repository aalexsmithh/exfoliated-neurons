import numpy as np
import matplotlib.pyplot as plt
import math

def main():
	minify("../data/train_x.bin")

def minify(dataset,samples=9,window_size=28):
	'''
	creates windows across the image and returns them in a python list
	dataset must be a .bin file
	'''

	data = np.fromfile(dataset, dtype='uint8')
	data = data.reshape((100000,60,60))

	if _check_inputs(samples,window_size) is False:
		print "function aborted."
	else:
		per_side = math.sqrt(samples)
		shift_val = (60 - window_size) / (per_side - 1)
		img_to_ret = []
		for i in range(0,int(per_side)):
			for j in range(0,int(per_side)):
				# print "grabbing on x from " + str(int(i*shift_val)) + " to " + str(int(i*shift_val+window_size))
				# print "grabbing on y from " + str(int(j*shift_val)) + " to " + str(int(j*shift_val+window_size))
				x = data[2][int(i*shift_val):int(i*shift_val+window_size),int(j*shift_val):int(j*shift_val+window_size)]
				img_to_ret.append(x)
		ret = np.asarray(img_to_ret)
		return ret

def _check_inputs(samples, window_size):
	if math.sqrt(samples) % 1 != 0:
		print math.sqrt(samples) % 1
		print "samples should be a square number."
		return False
	if window_size * math.sqrt(samples) < 60:
		print "this will not be optimal.",
		if window_size < 10:
			print "window size is too small.",
		if math.sqrt(samples) < 2:
			print "too few samples.",
		print
	if window_size * math.sqrt(samples) > 120:
		print "this will not be optimal.",
		if window_size > 40:
			print "window size is too large.",
		if math.sqrt(samples) > 5:
			print "too many samples.",
		print
	return True

if __name__ == '__main__':
	main()