import os
import sys
import json
import glob
import argparse
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import *
from sklearn.cluster import MiniBatchKMeans
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn import cross_validation
#
parser = argparse.ArgumentParser()
parser.add_argument("-f", "--featuresFileName", help="Output Features File", required="True")
args = parser.parse_args()
#
saved_features = {}

if (not os.path.isfile(args.featuresFileName)): 
		print args.featuresFileName + " doesn't exist!"
		sys.exit
#	

print "Loading from file..."
saved_features = {}
with open(args.featuresFileName, "r") as fin:
		for line in fin:
			data = line.split("\t")
			k = data[0]
			saved_features[k] = data[1].strip()
		#
	#
#
#
image_features = []
image_counter = 0
all_instance_filenames = []
all_instance_targets = []
for image in saved_features:
	image_counter+=1
	target = 1 if 'cat' in image else 0
	all_instance_filenames.append(image)
	all_instance_targets.append(target)
	#
	surf_data = []
	surf_lists = saved_features[image].split(":")
	for sl in surf_lists:
		this_data = map(float, sl.split(';'))
		surf_data.append(this_data)
	#	
	image_features.append(surf_data)
#
print image_counter, " images loaded..."
#

X_train_surf_features = np.concatenate(image_features)

# Clusters
n_clusters = 300
print 'Clustering', len(X_train_surf_features), 'features'
estimator = MiniBatchKMeans(n_clusters=n_clusters)
estimator.fit_transform(X_train_surf_features)

x_data = []
for instance in image_features:
	clusters = estimator.predict(instance)
	features = np.bincount(clusters)
	if len(features) < n_clusters:
		features = np.append(features, np.zeros((1, n_clusters-len(features))))
	#
	x_data.append(features)


#
#train_len = int(len(all_instance_filenames) * .70)
#X_train	= x_data[:train_len]
#X_test  = x_data[train_len:]
#print "Using ", train_len, " images to train the models" 
#
#y_train = all_instance_targets[:train_len]
#y_test  = all_instance_targets[train_len:]
#
#
#clf = tree.DecisionTreeClassifier()
clf = RandomForestClassifier(max_depth=5, n_estimators=20, max_features=1)
#clf = RandomForestClassifier()												
#clf = AdaBoostClassifier()
#clf = SVC(kernel="linear", C=0.025)
#clf = SVC(gamma=2, C=1)
#clf = GaussianNB()

scores = cross_validation.cross_val_score(clf, x_data, all_instance_targets, cv=5)
print ""
print "CV Scores....: ", scores
print "CV Mean score: ", sum(scores)/(len(scores)*1.0)


#---------------------------------------------------------
# Por si queremos testear sin CV
#clf.fit(X_train, y_train)
#predictions = clf.predict(X_test)
#print classification_report(y_test, predictions)
#print 'Precision: ', precision_score(y_test, predictions)
#print 'Recall: ', recall_score(y_test, predictions)
#print 'Accuracy: ', accuracy_score(y_test, predictions)

