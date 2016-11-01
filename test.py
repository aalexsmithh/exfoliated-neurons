# -*- coding: utf-8 -*-
"""
Created on Mon Oct 31 22:50:50 2016

@author: Sandy Wong
"""

import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn import linear_model, datasets
from sklearn.model_selection import cross_val_score
from sklearn import svm
import pickle
import csv
import load
'''
iris = datasets.load_iris()
X = iris.data[:, :2]  # we only take the first two features.
Y = iris.target
print ("X ",X.shape," y: ", y.shape)

'''
data = load.load("X.pkl")
y = []
with open('train_y.csv', newline='') as f_in:
    csvreader = csv.reader(f_in)
    for row in csvreader:
        y.append(row[1])
    
X = np.matrix(data)[:,:]
Y = np.array(y)
Y = Y.transpose()[:]

print ("X: ",X.shape)
print ("Y: ", Y.shape)


c=1
logreg = linear_model.LogisticRegression(C=1e5)
svc = svm.SVC(kernel='linear', C=c)
rbf_svc = svm.SVC(kernel='rbf', gamma=0.7, C=c)
poly_svc = svm.SVC(kernel='poly', degree=3, C=c)
lin_svc = svm.LinearSVC(C=c)

classifiers = []
classifiers.append(logreg)
classifiers.append(svc)
classifiers.append(rbf_svc)
classifiers.append(poly_svc)
classifiers.append(lin_svc)

cNames = ["logreg","svc","rbf_svc","poly_svc","lin_svc"]

index = 0
for clf in classifiers:
    print (cNames[index]) 
    index+=1
    scores = cross_val_score(clf, X, Y, cv=3)
    for score in scores:
        print (score)
    mean = scores.mean()
    print (mean, "\n")

