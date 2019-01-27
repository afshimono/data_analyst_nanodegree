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
import numpy as np
print('Started.')
### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###

features_train = features_train[:len(features_train)/100] 
labels_train = labels_train[:len(labels_train)/100] 

clf = svm.SVC(kernel='rbf',C=10000)
print('Starting training.')
t0 = time()
clf.fit(features_train, labels_train)  
print "training time:", round(time()-t0, 3), "s"

t0 = time()
prediction = clf.predict(features_test)
print "prediction time:", round(time()-t0, 3), "s"

results = np.asarray(prediction) - np.asarray(labels_test)
target_hit = 0
for num in results:
	if num == 0:
		target_hit += 1
accuracy = float(target_hit) / len(results)
print(accuracy)

print('Element 10:  '+str(prediction[10]))
print('Element 26:  '+str(prediction[26]))
print('Element 50:  '+str(prediction[50]))

#########################################################


