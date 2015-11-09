import os
import sys
import json
import glob
import argparse
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
saved_features = json.load(open(args.featuresFileName))
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
	hist = []
	hist = map(float, saved_features[image].split(';'))
  	image_features.append(hist)
#
print image_counter, " images loaded..."
#
#train_len = int(len(all_instance_filenames) * .70)
#X_train	= image_features[:train_len]
#X_test  = image_features[train_len:]
#print "Using ", train_len, " images to train the models" 
#
#y_train = all_instance_targets[:train_len]
#y_test  = all_instance_targets[train_len:]
#
#
clf = tree.DecisionTreeClassifier()
#clf = RandomForestClassifier(max_depth=5, n_estimators=20, max_features=1)
#clf = RandomForestClassifier()												
#clf = AdaBoostClassifier()
#clf = SVC(kernel="linear", C=0.025)
#clf = SVC(gamma=2, C=1)
#clf = GaussianNB()

scores = cross_validation.cross_val_score(clf, image_features, all_instance_targets, cv=5)
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

