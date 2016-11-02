from sklearn.neural_network import MLPClassifier
from sklearn import preprocessing
import csv
import numpy as np

def main():
	data = np.fromfile('../train_x.bin', dtype='uint8')
	data = data.reshape((100000,3600))
	Y = open_csv("data/train_y.csv")

	scaler = preprocessing.StandardScalar().fit(data[:10000])
	X_train = scaler.transform(data[:10000])
	X_test = scaler.transform(data[10000:])

	clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5, 2), random_state=1)
	clf.fit(X_train,Y[:10000])
	clf.score(X_test,Y[10000:])


def open_csv(filename):
	with open(filename, 'rb') as csvfile:
		ret = []
		csv_in = csv.reader(csvfile, delimiter=',', quotechar='"')
		for line in csv_in:
			ret.append(line[1])
		return ret[1:]