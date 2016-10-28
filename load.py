try:
	import cPickle as pickle
except Exception as e:
	import pickle

def load(filename):
	return pickle.load(open(filename,'rb'))
