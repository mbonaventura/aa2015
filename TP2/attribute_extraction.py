import sys
import numpy as np
import glob
import cv2
from sklearn import cross_validation
from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import *
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import MiniBatchKMeans
from sklearn.naive_bayes import GaussianNB
import mahotas as mh
from mahotas.features import surf

def getAttributes_colors(filePattern):
	all_instance_filenames = []
	targets_data = []
	for f in glob.glob(filePattern):
		target = 1 if 'cat' in f else 0
		all_instance_filenames.append(f)
		targets_data.append(target)
		#print f, target

	x_data = []
	counter = 0
	for f in all_instance_filenames:
		counter+=1
		print 'Reading image: ', counter, f
		image = cv2.imread(f, 0)
		#equimage = cv2.equalizeHist(image)
		#this_histo, bins = np.histogram(equimage.flatten(), 256, [0, 256])
		this_histo, bins = np.histogram(image.flatten(), 256, [0, 256])
		x_data.append(this_histo)

	return x_data, targets_data

def getAttributes_surf(filePattern):
	targets_data = []
	surf_features = []
	counter = 0
	for f in glob.glob(filePattern):
		counter+=1
		print 'Reading image: ', counter, f

		target = 1 if 'cat' in f else 0
		targets_data.append(target)
		
		image = mh.imread(f, as_grey=True)
		surf_features.append(surf.surf(image)[:, 5:])

	X_train_surf_features = np.concatenate(surf_features)
	
	# Clusters
	n_clusters = 300
	print 'Clustering', len(X_train_surf_features), 'features'
	estimator = MiniBatchKMeans(n_clusters=n_clusters)
	estimator.fit_transform(X_train_surf_features)

	x_data = []
	for instance in surf_features:
		clusters = estimator.predict(instance)
		features = np.bincount(clusters)
		if len(features) < n_clusters:
			features = np.append(features, np.zeros((1, n_clusters-len(features))))

		x_data.append(features)

	return x_data, targets_data