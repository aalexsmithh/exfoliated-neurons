import numpy as np
import scipy.misc as disp
import cv2, sklearn.cluster, cPickle, time, csv, load,sklearn.linear_model

def main():
	X = load.load("X1.pkl")
	Y = open_csv("data/train_y.csv")
	lr = sklearn.linear_model.LogisticRegression()
	lr.fit(X[0:70000],Y[0:70000])
	print lr.score(X[70000:],Y[70000:])


def get_clust():
	already_compute_feats = False
	a = time.time()
	print "Opening data..."
	data = np.fromfile('data/train_x.bin', dtype='uint8')
	data = data.reshape((100000,60,60))
	print "Opening clusterer..."
	clusters = open_to('sift_50_mini.pkl')
	print "Getting features..."
	if already_compute_feats:
		features = open_to('feats.pkl')
	else:
		features = feature_extract(data[0:100000],'sift')
	print "Assembling vectors..."
	X = make_vectors(features,clusters,50)
	print "Saving..."
	save_to(X,'X1.pkl')

	print "Total computation time was", time.time() - a

def run_clust():
	already_compute_feats = False
	a = time.time()
	if not already_compute_feats:
		print "Opening data..."
		data = np.fromfile('data/train_x.bin', dtype='uint8')
		data = data.reshape((100000,60,60))
		print "Getting features..."
		features = feature_extract(data[0:100000],'sift')
		save_to(features,'feats.pkl')
	else:
		print "Getting features..."
		features = open_to('feats.pkl')
	print "Creating vocab...",
	vocab = create_vocab(features,128)
	print "\t" + str(len(vocab)) + " features assembled"
	print "Clustering..."
	# clusters = sklearn.cluster.KMeans(n_clusters=50,verbose=0)
	clusters = sklearn.cluster.MiniBatchKMeans(50)
	clusters.fit(vocab)
	print "Saving..."
	save_to(clusters,"sift_50_mini.pkl")

	print "Total computation time was", time.time() - a

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
	run_clust()
	get_clust()
	main()
