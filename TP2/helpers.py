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



def gridSearch(X_train, y_train):
	# Set the parameters by cross-validation
    
    #tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
	#                     'C': [1, 10, 100, 1000]},
	#                    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]
	#estimator = SVC(C=1)
	scores = ['precision'] # ['accuracy', 'adjusted_rand_score', 'average_precision', 'f1', 'f1_macro', 'f1_micro', 'f1_samples', 'f1_weighted', 'log_loss', 'mean_absolute_error', 'mean_squared_error', 'median_absolute_error', 'precision', 'precision_macro', 'precision_micro', 'precision_samples', 'precision_weighted', 'r2', 'recall', 'recall_macro', 'recall_micro', 'recall_samples', 'recall_weighted', 'roc_auc']
	estimators = ['SVC', 'RandomForest', 'LogisticRegression', 'DecisionTree', 'AdaBoost', 'GaussianNB'] #, 
	
	
	bestEstimators=[]
	for estimatorName in estimators:
		estimator, tuned_parameters = getSearch(estimatorName)
		
		print("")
		print ("# Finding best parameters for %s:" % estimatorName)
		print ("----------------------------")
		for scoreName in scores:
			# Do the grid search
		    clf = GridSearchCV(estimator, tuned_parameters, cv=5,
		                       scoring='%s' % scoreName)
		    clf.fit(X_train, y_train)

		    # Add the best estimator to the result
		    bestEstimators.append((clf.best_estimator_, clf.best_score_))

		    print("\tBest parameters: %s" % clf.best_estimator_)
		    print("\tBest score (%s): %s" % (scoreName, clf.best_score_))	
		    print("")	    
		    
		    print("Grid scores on development set:")
		    print("")
		    for params, mean_score, grid_scores in clf.grid_scores_:
		        print("%0.3f (+/-%0.03f) for %r"
		              % (mean_score, grid_scores.std() * 2, params))
		    print("")		    
	return bestEstimators

def getSearch(estimatorName):
	estimators = {} 
	estimators["SVC"] = (
							SVC(C=1), 
						    [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4, 2], 'C': [1, 10, 100, 1000]},
	                         {'kernel': ['linear'], 'C': [0.0025, 1, 5, 100, 1000]}]	
	                    )
	estimators["RandomForest"] = (
								   RandomForestClassifier(), 
						           [{'max_depth': [2, 5, 10], 'n_estimators': [1, 10, 5], 'max_features': [1, 3, 6, 10]}]
						         )
	estimators["LogisticRegression"] = (
								   			LogisticRegression(), 
						           			[{'penalty': ['l2'], 'C': [0.001, 0.01, 0.005, 0.0005]}]
						         		)
	estimators["DecisionTree"] = (
								   tree.DecisionTreeClassifier(), 
						           [{'max_depth': [5, 10], 'max_leaf_nodes': [50, 100]}]
						         )
	estimators["AdaBoost"] = (
								AdaBoostClassifier(), 
						        [{'n_estimators': [20, 50, 100], 'algorithm': ['SAMME.R']}]
						     )
	estimators["GaussianNB"] = (
								GaussianNB(), 
						        [{}]
						     )

	if estimators.has_key(estimatorName):
		return estimators[estimatorName]
	else:
		print "Unexpected estimator name %s" % estimatorName	
		print "available estimators: ", estimators.keys()


	raise ValueError('Unexpected estimator name')

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