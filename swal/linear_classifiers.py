import numpy as np
import scipy.misc as disp
import sklearn.cluster, cPickle, time, csv, sklearn.linear_model, cv2
from sklearn import linear_model, svm
from sklearn.model_selection import cross_val_score, train_test_split, KFold

def main():

	#define the linear classifiers
	logreg = linear_model.LogisticRegression(C=1e5,n_jobs=-1,solver='sag')
	lin_svc = svm.SVC(kernel='linear')
	rbf_svc = svm.SVC(kernel='rbf', gamma=0.7)
	poly_svc = svm.SVC(kernel='poly', degree=3)

	#define the different options for testing
	feature_types = ['sift','surf']
	k_vals = [20,50]
	classifiers = [logreg,rbf_svc,poly_svc,lin_svc]
	cNames = ["Logistic Regression", "Rbf SVM", "Polynomial SVM", "Linear SVM"]

	try:
		#open and split data into test and train sets
		data = np.fromfile("train_x.bin", dtype='uint8')
		data = data.reshape((100000,60,60))
		labels = open_csv("train_y.csv")

		data_train = data[0:70000]
		data_test = data[70000:]
		Y_train = labels[0:70000]
		Y_test = labels[70000:]

		print "WARNING: This will take a while. Expect 4 or more hours."

		for feat_to_test in feature_types:
			#extract features for the train and test set
			features_train = feature_extract(data_train,feat_to_test)
			features_test = feature_extract(data_test,feat_to_test)
			vocab_train = create_vocab(features_train,128)

			for k in k_vals:
				#cluster into different k values
				clusters = sklearn.cluster.MiniBatchKMeans(k)
				clusters.fit(vocab_train)

				X_train = make_vectors(features_train,clusters,k)
				X_test = make_vectors(features_test,clusters,k)

				for i, classifier in enumerate(classifiers):
					#fit for each k and feature type onto each model
					classifier.fit(X_train,Y_train)
					score = classifier.score(X_test,Y_test)
					print cNames[i] + ' with ' + feat_to_test + ' features, clustered into ' + str(k) + ' groups gives ' + str(score) + " accuracy on 30% of the training data "
				print
			print

	except IOError as e:
		print "Files must be in the same folder as this script."

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

if __name__ == '__main__':
	main()