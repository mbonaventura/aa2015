import sys
import numpy as np
from sklearn import cross_validation
#import mahotas as mh
#from mahotas.features import surf
#from sklearn.linear_model import LogisticRegression
from sklearn.metrics import *
#from sklearn.cluster import MiniBatchKMeans
import glob
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
#from sklearn.svm import SVC
#from sklearn.naive_bayes import GaussianNB
import cv2


def oscuro(value):
	return '0' if value>127 else '1'
#-----------------------------------------------------------------------------
all_instance_filenames = []
targets_data = []
for f in glob.glob('./img2/*.jpg'):
	target = 1 if 'cat' in f else 0
	all_instance_filenames.append(f)
	targets_data.append(target)
	#print f, target

x_data = []
counter = 0
for f in all_instance_filenames:
	counter+=1
	#print 'Reading image: ', counter, f
	image = cv2.imread(f, 0)
	#equimage = cv2.equalizeHist(image)
	#this_histo, bins = np.histogram(equimage.flatten(), 256, [0, 256])
	this_histo, bins = np.histogram(image.flatten(), 256, [0, 256])
	x_data.append(this_histo)
#

X_train, X_test, y_train, y_test = cross_validation.train_test_split(x_data, targets_data, test_size=0.3, random_state=0)

#print X_train[0]

clf = RandomForestClassifier()

clf.fit(X_train, y_train)

predictions = clf.predict(X_test)

print classification_report(y_test, predictions)
print 'Precision: ', precision_score(y_test, predictions)
print 'Recall: ', recall_score(y_test, predictions)
print 'Accuracy: ', accuracy_score(y_test, predictions)
	