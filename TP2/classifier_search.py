import sys
import os
import numpy as np
import glob
import cv2
from sklearn import cross_validation
from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import *
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import MiniBatchKMeans
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
import mahotas as mh
from mahotas.features import surf
from sklearn.externals import joblib

def saveEstimator(estimator, score, directory):
	if not os.path.exists(directory):
		print("created directory: %s " % directory)	
		os.makedirs(directory)

	estimatorName = ("%s" % estimator).split('(')[0]
	baseFileName = os.path.join(directory, estimatorName)

	#save score
	with open("%s.score" % baseFileName, "w") as text_file:
		text_file.write("%f" % score)	

	# write estimator paramas	
	with open("%s.params" % baseFileName, "w") as text_file:
		text_file.write("%s" % estimator)

	#save estimator
	joblib.dump(estimator, "%s.pkl" % baseFileName) 

	print("estimator saved to file: %s " % baseFileName)

def loadEstimators(path):
	estimators = []
	for f in glob.glob(os.path.join(path, '*.pkl')):
		estimator = joblib.load(f) 

		baseFileName = os.path.splitext(f)[0]
		with open ("%s.score" % baseFileName, "r") as scoreFile:
			score = float(scoreFile.readline())
			

    	estimators.append((estimator, score))
	return estimators


def gridSearch(X_train, y_train, estimators, featureSet, n_jobs=1):
	# Set the parameters by cross-validation
    
    #tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
	#                     'C': [1, 10, 100, 1000]},
	#                    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]
	#estimator = SVC(C=1)
	scores = ['precision'] # ['accuracy', 'adjusted_rand_score', 'average_precision', 'f1', 'f1_macro', 'f1_micro', 'f1_samples', 'f1_weighted', 'log_loss', 'mean_absolute_error', 'mean_squared_error', 'median_absolute_error', 'precision', 'precision_macro', 'precision_micro', 'precision_samples', 'precision_weighted', 'r2', 'recall', 'recall_macro', 'recall_micro', 'recall_samples', 'recall_weighted', 'roc_auc']
	
	bestEstimators=[]
	for estimatorName in estimators:
		estimator, tuned_parameters = getSearch(estimatorName)
		
		print("")
		print ("# Finding best parameters for %s:" % estimatorName)
		print ("----------------------------")
		for scoreName in scores:
			# Do the grid search
		    clf = GridSearchCV(estimator, tuned_parameters, cv=5, 
		                       scoring='%s' % scoreName,verbose=10, n_jobs=n_jobs)
		    clf.fit(X_train, y_train)

		    # Add the best estimator to the result
		    bestEstimators.append((clf.best_estimator_, clf.best_score_))

		    # save estimator
		    saveEstimator(clf.best_estimator_, clf.best_score_, './estimators/%s' % featureSet)  

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
						     #[{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4, 2], 'C': [1, 10, 100, 1000]},
	                         #{'kernel': ['linear'], 'C': [0.0025, 1, 5, 100, 1000]}]	
	                         [{'kernel': ['rbf'], 'gamma': [1e-3], 'C': [1, 100]},
	                         {'kernel': ['linear'], 'C': [1, 5]}]	
	                    )
	estimators["RandomForest"] = (
								   RandomForestClassifier(), 
						           [{'n_estimators': [20], 'max_features': ['auto', 5, 20, 100], 'max_depth': [10, 50, 100, 300]}]
						           #[{'n_estimators': [20], 'max_features': ['auto', 5, 20, 100, 200], 'max_depth': [10, 100, 300]}]
						         )
	estimators["ExtraTrees"] = (
								ExtraTreesClassifier(),
								[{'n_estimators': [1, 5, 10],  'max_features':['auto', 1, 5, 10], 'max_depth': [None, 5, 10], 'min_samples_split':[1], 'random_state':[0]}]
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
	estimators["GradientBoosting"] = (
								GradientBoostingClassifier(), 
						        [{'n_estimators': [20, 100, 150], 'learning_rate': [1.0, 0.1], 'max_depth':[3, 10], 'loss':['deviance']}]
						     )	
	estimators["GaussianNB"] = (
								GaussianNB(), 
						        [{}]
						     )
	estimators["Bagging"] = (
								BaggingClassifier(),
								[{'base_estimator': [KNeighborsClassifier(), tree.DecisionTreeClassifier()],
								  'n_estimators': [5, 10], 'max_features':[0.5], 'max_samples':[0.5]}]
							)
	


	if estimators.has_key(estimatorName):
		return estimators[estimatorName]
	else:
		print "Unexpected estimator name %s" % estimatorName	
		print "available estimators: ", estimators.keys()


	raise ValueError('Unexpected estimator name')