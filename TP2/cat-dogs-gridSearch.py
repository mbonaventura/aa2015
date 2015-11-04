import sys
import numpy as np
from sklearn import cross_validation
from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report

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
from helpers import *


# Get attributes and class for the images
#x_data, targets_data = getAttributes_surf('./img20/*.jpg')
x_data, targets_data = getAttributes_colors('./img200/*.jpg')

# Separate in train & test
X_train, X_test, y_train, y_test = cross_validation.train_test_split(x_data, targets_data, test_size=0.3, random_state=0)

#Perform a grid search to find best method and parameters
bestEstimators = gridSearch(X_train, y_train)

# print the result for the best estimators
bestbest_estimator, bestbest_score = bestEstimators[0]
bestbest_score = 0
for estimator, score in bestEstimators:
	predictions = estimator.predict(X_test)

	print("")
	print("Testing %s with TEST data:" % estimator)		    
	print("========================================================================================")
	print classification_report(y_test, predictions)
	print 'Precision: ', precision_score(y_test, predictions)
	print 'Recall: ', recall_score(y_test, predictions)
	print 'Accuracy: ', accuracy_score(y_test, predictions)
	print 'Expected (precision): ', score

	if(bestbest_score < precision_score(y_test, predictions)):
		bestbest_score = precision_score(y_test, predictions)
		bestbest_estimator = estimator


	
print("========================================================================================")
print("========================================================================================")
print("The very best best estimator is: (Precision %f )" % bestbest_score)
print(bestbest_estimator)



	
