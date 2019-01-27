#!/usr/bin/python


"""
    Starter code for the evaluation mini-project.
    Start by copying your trained/tested POI identifier from
    that which you built in the validation mini-project.

    This is the second step toward building your POI identifier!

    Start by loading/formatting the data...
"""

import pickle
import sys
import numpy

sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "r") )

### add more features to features_list!
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)



### your code goes here 

reshape_labels = numpy.reshape(labels,(len(labels),1))
reshape_features = numpy.reshape(features,(len(features),1))

from sklearn.cross_validation import train_test_split
feature_train, feature_test, labels_train, labels_test = train_test_split(reshape_features,
	reshape_labels,test_size=0.3, random_state=42)

from sklearn import tree
clf = tree.DecisionTreeClassifier()
clf=clf.fit(feature_train,labels_train)


print('Prediction:  ')
reshaped_prediction = numpy.reshape(clf.predict(feature_test),(len(clf.predict(feature_test)),1))
print(reshaped_prediction)
print('Real: ')
print(labels_test)

from sklearn.metrics import recall_score
print('Recall score:  '+ str(recall_score(labels_test,reshaped_prediction)))

from sklearn.metrics import precision_score
print('Precision score:  '+ str(precision_score(labels_test,reshaped_prediction)))