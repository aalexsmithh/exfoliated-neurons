import numpy as np
import scipy.misc as disp
from multiprocessing.pool import ThreadPool
import cv2, sklearn.cluster, threading, cPickle, time

def main():
	X = load.load("X.pkl")
	print X

def get_clust():
	a = time.clock()
	print "Opening data..."
	data = np.fromfile('data/train_x.bin', dtype='uint8')
	data = data.reshape((100000,60,60))
	print "Opening clusterer..."
	clusters = open_to('surf_50_full.pkl')
	print "Getting features..."
	features = feature_extract(data[0:100000],'surf')
	print "Assembling vectors..."
	X = make_vectors(features,clusters,50)
	print "Saving..."
	save_to(X,'X.pkl')

	print "Total computation time was", time.clock() - a

def run_clust():
	a = time.clock()
	print "Opening data..."
	data = np.fromfile('data/train_x.bin', dtype='uint8')
	data = data.reshape((100000,60,60))
	print "Getting features..."
	features = feature_extract(data[0:100000],'surf')
	print "Creating vocab...",
	vocab = create_vocab(features,128)
	print "\t" + str(len(vocab)) + " features assembled"
	print "Clustering..."
	clusters = sklearn.cluster.KMeans(n_clusters=50,verbose=0)
	# clusters = sklearn.cluster.MiniBatchKMeans(50)
	clusters.fit(vocab)
	print "Saving..."
	save_to(clusters,"clust.pkl")

	print "Total computation time was", time.clock() - a

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

def get_orb_features(data,display=False):
	features = []
	orb = cv2.ORB_create()
	for img in data:
		kp = orb.detect(img,None)
		kp, des = orb.compute(img, kp)
		features.append(des)
		if display:
			img2 = cv2.drawKeypoints(img,kp,None,(255,0,0),4)
			show_img(img2,("%d" %len(kp)))
	return features

def export_csv(out,filename): #use .gz to auto export zipped file
	j = out.reshape((100000,3600))
	np.savetxt(filename,j,fmt='%d',delimiter=",")

def save_to(data, filename):
	cPickle.dump(data,open(filename,'wb'))

def open_to(filename):
	return cPickle.load(open(filename,'rb'))

def import_csv(filename):
	data = np.loadtxt(filename,dtype='unit8',delimiter=',')

def show_img(img,caption):
	cv2.imshow(caption,img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()


if __name__ == '__main__':
	main()
