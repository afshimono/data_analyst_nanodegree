#!/usr/bin/python


"""
    Starter code for the validation mini-project.
    The first step toward building your POI identifier!

    Start by loading/formatting the data

    After that, it's not our code anymore--it's yours!
"""

import pickle
import sys
import numpy

sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "r") )

### first element is our labels, any added elements are predictor
### features. Keep this the same for the mini-project, but you'll
### have a different feature list when you do the final project.
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)

#print(len(labels))


### it's all yours from here forward!  
reshape_labels = numpy.reshape(labels,(len(labels),1))
reshape_features = numpy.reshape(features,(len(features),1))
#print(reshape_features)


from sklearn.cross_validation import train_test_split
feature_train, feature_test, labels_train, labels_test = train_test_split(reshape_features,
	reshape_labels,test_size=0.3, random_state=42)

from sklearn import tree
clf = tree.DecisionTreeClassifier()
clf=clf.fit(feature_train,labels_train)

print(clf.score(feature_test,labels_test))
