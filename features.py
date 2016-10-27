import numpy as np
import scipy.misc as disp
from multiprocessing.pool import ThreadPool
import cv2, sklearn.cluster, threading


def main():
	X_DIM, Y_DIM = 60,60
	data = np.fromfile('data/train_x.bin', dtype='uint8')
	data = data.reshape((100000,60,60))

	print "Getting features..."
	# features = get_surf_features(data[0:100000])
	features = feature_extract(data[0:10000],'surf')
	print "Creating vocab..."
	vocab = create_vocab(features,128)
	print vocab.shape
	print "Clustering..."
	clusters = sklearn.cluster.k_means(vocab, 50, n_jobs=-2)
	print clusters

def create_vocab(data,feats):
	vocab = np.empty((feats,0),float)
	for line in data:
		if line is not None:
			for des in line:
				vocab = np.append(vocab,des)
	return vocab.reshape(vocab.shape[0]/feats,feats)

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

def import_csv(filename):
	data = np.loadtxt(filename,dtype='unit8',delimiter=',')

def show_img(img,caption):
	cv2.imshow(caption,img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()


if __name__ == '__main__':
	main()