# -*- coding: utf-8 -*-
"""
Created on Fri Nov  4 01:33:24 2016

@author: Sandy Wong
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model, datasets
import csv
from sklearn.model_selection import cross_val_score

x = np.fromfile('train_x.bin', dtype='uint8')
print (x.shape)
x = x.reshape(100000,3600)
#x = x[:500, :2] #first 2 features
y_raw = []

i = 0
with open('train_y.csv', 'r') as f_in:
    csvreader = csv.reader(f_in)
    for row in csvreader:
        y_raw.append(int(row[1]))
        
y = np.array(y_raw)
#y = y[:500]
print (x.shape," ",y.shape)

logreg = linear_model.LogisticRegression(C=1e5)

f = open('results_logreg.txt', 'w')

print ("Logistic Regression on 3600 bytes raw features 3 fold cross validation")
f.write("Logistic Regression on 3600 bytes raw features 3 fold cross validation\n")
scores = cross_val_score(logreg, x, y, cv=3)
for score in scores:
    print (score)
    f.write(str(score)+"\n")
print ("Mean: ", scores.mean())
f.write("Mean: "+str(scores.mean())+"\n")
f.close()


