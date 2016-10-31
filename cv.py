# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 18:12:00 2016

@author: Sandy Wong
Cross Validation for Project 3 COMP 551
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model, datasets
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn import svm
import load
from sklearn.model_selection import KFold
import pickle
import csv

cNames=[]
    
def printscore(array):
    for i in range(len(array)):
        print ("\n", cNames[i],"\n")
        for j in range(len(array[i])):
            print (array[i][j]," ")

#import data TEMP
iris = datasets.load_iris()
data = load.load("X.pkl")
#X = iris.data[:, :2]  # we only take the first two features.
#Y = iris.target

Y = []
with open('train_y.csv', newline='') as f_in:
    csvreader = csv.reader(f_in)
    for row in csvreader:
        Y.append(row[1])
    
X = np.matrix(data)
y = np.matrix(Y)
y = y.transpose()

print ("X: ",X.shape)
print ("y: ", y.shape)


#classifiers
logreg = linear_model.LogisticRegression(C=1e5)
C=1
svc = svm.SVC(kernel='linear', C=C)
rbf_svc = svm.SVC(kernel='rbf', gamma=0.7, C=C)
poly_svc = svm.SVC(kernel='poly', degree=3, C=C)
lin_svc = svm.LinearSVC(C=C)

#do classification
kf = KFold(n_splits=3)
classifiers = [logreg,svc,rbf_svc,poly_svc,lin_svc]
cNames = ["Logistic Regression", "SVC", "Rbf SVC", "Polynomial SVC", "Linear SVC"]
scores = [[],[],[],[],[]]

scores2 = []

'''
for train_index, test_index in kf:
    #print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    for i in range(len(classifiers)):
        clf = classifiers[i]
        clf.fit(X_train,y_train)
        name = cNames[i]
        score = clf.score(X_test, y_test)
        scores[i].append(score)
'''

count = 0

for clf in classifiers:
    print ("cross validating...\n")
    score = cross_val_score(clf, X, y, cv=kf, n_jobs=-1)
    count = count+1
    print("cross validation number: ", count)
    scores2.append(score)
    
printscore(scores2)

        
        


