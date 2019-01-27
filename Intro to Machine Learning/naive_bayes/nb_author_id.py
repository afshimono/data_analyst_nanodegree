#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 1 (Naive Bayes) mini-project. 

    Use a Naive Bayes Classifier to identify emails by their authors
    
    authors and labels:
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess
from sklearn.naive_bayes import GaussianNB
import numpy as np

### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()


#########################################################
### your code goes here ###
gnb = GaussianNB()
t0 = time()
gnb.fit(features_train, labels_train)
print "training time:", round(time()-t0, 3), "s"

t0 = time()
y_pred = gnb.predict(features_test)
print "predict time:", round(time()-t0, 3), "s"

result_array = np.asarray(y_pred) - np.asarray(labels_test)
target_hit = 0
for num in result_array:
	if num == 0:
		target_hit += 1
accuracy = float(target_hit)/len(result_array)
print(accuracy)

#########################################################


