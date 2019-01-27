#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 3 (decision tree) mini-project.

    Use a Decision Tree to identify emails from the Enron corpus by author:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess
from sklearn import tree
import numpy as np




### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###
#print(str(len(features_train[0])))
clf = tree.DecisionTreeClassifier(min_samples_split=40)
clf = clf.fit(features_train, labels_train)
predict = clf.predict(features_test)
result = np.asarray(labels_test) - np.asarray(predict)
target_hit = 0
for num in result:
	if num == 0:
		target_hit += 1
accuracy = float(target_hit) / len(result)
print("Accuracy:  "+str(accuracy))

#########################################################


