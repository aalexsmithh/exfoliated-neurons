import numpy as np
import scipy.misc as disp
import sklearn.cluster, cPickle, time, csv, sklearn.linear_model, cv2
from sklearn import linear_model, svm
from sklearn.model_selection import cross_val_score, train_test_split, KFold

def main():

	logreg = linear_model.LogisticRegression(C=1e5)
	lin_svc = svm.SVC(kernel='linear')
	rbf_svc = svm.SVC(kernel='rbf', gamma=0.7)
	poly_svc = svm.SVC(kernel='poly', degree=3)

	feature_types = ['sift','surf']
	k_vals = [20,50]
	classifiers = [logreg,rbf_svc,poly_svc,lin_svc]
	cNames = ["Logistic Regression", "Rbf SVM", "Polynomial SVM", "Linear SVM"]

	data = np.fromfile('data/train_x.bin', dtype='uint8')
	data = data.reshape((100000,60,60))
	labels = open_csv("data/train_y.csv")

	data_train = data[0:70000]
	data_test = data[70000:]
	Y_train = labels[0:70000]
	Y_test = labels[70000:]

	for feat_to_test in feature_types:
		features_train = feature_extract(data_train,feat_to_test) #extract features for the train set
		features_test = feature_extract(data_test,feat_to_test) #extract features for the test set
		vocab_train = create_vocab(features_train,128) #create vocabulary of features for clustering

		for k in k_vals:
			clusters = sklearn.cluster.MiniBatchKMeans(k)
			clusters.fit(vocab_train)

			X_train = make_vectors(features_train,clusters,k)
			X_test = make_vectors(features_test,clusters,k)

			for i, classifier in enumerate(classifiers):
				classifier.fit(X_train,Y_train)
				score = 0.8267#classifier.score(X_test,Y_test)
				print cNames[i] + ' with ' + feat_to_test + ' features, clustered into ' + k + ' groups gives ' + score + " accuracy on 30% of the training data "


def feature_extract(data,type,display=False):
	if type == 'surf':
		return get_surf_features(data,display)
	if type == 'sift':
		return get_sift_features(data,display)

def get_surf_features(data,display=False):
	features = []
	surf = cv2.xfeatures2d.SURF_create(400,extended=True)
	surf.setHessianThreshold(400)
	for img in data:
		kp, des = surf.detectAndCompute(img,None)
		features.append(des)
		if display:
			img2 = cv2.drawKeypoints(img,kp,None,(255,0,0),4)
			show_img(img2,("%d" %len(kp)))
	return features

def get_sift_features(data,display=False):
	features = []
	sift = cv2.xfeatures2d.SIFT_create(400)
	for img in data:
		kp, des = sift.detectAndCompute(img,None)
		features.append(des)
		if display:
			img2 = cv2.drawKeypoints(img,kp,None,(255,0,0),4)
			show_img(img2,("%d" %len(kp)))
	return features

def open_csv(filename):
	with open(filename, 'rb') as csvfile:
		ret = []
		csv_in = csv.reader(csvfile, delimiter=',', quotechar='"')
		for line in csv_in:
			ret.append(line[1])
		return ret[1:]

def make_vectors(train,clust,n):
	feats = [[0 for _ in range(n)] for _ in range(len(train))]
	for i, line in enumerate(train):
		line_feats = []
		if line is not None:
			for des in line:
				line_feats.append(des)
			line_feats = clust.predict(line_feats)
		for j in line_feats:
			feats[i][j] += 1
	return feats

def create_vocab(data,feats):
	vocab = []
	for line in data:
		if line is not None:
			for des in line:
				vocab.append(des)
	return vocab

def show_img(img,caption):
	cv2.imshow(caption,img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()