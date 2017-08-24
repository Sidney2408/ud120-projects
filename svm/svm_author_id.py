#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess
from sklearn import svm

### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
print "Processing data"
features_train, features_test, labels_train, labels_test = preprocess()
clf = svm.SVC(kernel='linear', gamma = 1.0)
print "Fitting data"
clf.fit(features_train, labels_train)
print "predicting data"
pred = clf.predict(features_test)

from sklearn.metrics import accuracy_score
acc = accuracy_score(pred, labels_test)

print acc




#########################################################
### your code goes here ###


#########################################################


