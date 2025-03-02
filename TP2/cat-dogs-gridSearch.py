import sys
import numpy as np
from sklearn import cross_validation

from sklearn.metrics import classification_report
from sklearn.metrics import *
from sklearn.ensemble import VotingClassifier
import glob
import cv2
from attribute_extraction import *
from classifier_search import *
import argparse

#### PARAMETERS  ######
parser = argparse.ArgumentParser()
parser.add_argument("-t", "--threads", help="Numbers to use in the grid search", required="True")
parser.add_argument("-f", "--featureName", help="Features to use", required="True")
parser.add_argument("-e", "--estimators", help="estimators to use", required="True")
args = parser.parse_args()

threads = int(args.threads)
featureSet =  args.featureName #'surf-c100' # 'bp-r5' 'hara-img200'

# FEW estimators
estimators = []
if args.estimators == "few":
	estimators = ['GaussianNB'] 
elif args.estimators == "some":
	# SOME estimators
	estimators = ['RandomForest', 'LogisticRegression', 'DecisionTree', 'AdaBoost', 'GaussianNB']
elif args.estimators == "all":
	# ALL estimators
	estimators = ['GradientBoosting', 'ExtraTrees', 'Bagging', 'RandomForest', 'LogisticRegression', 'DecisionTree', 'AdaBoost', 'GaussianNB', 'SVC'] 
else :
	print "ERROR -e must be 'few', 'some' or 'all'."
	sys.exit()


print "Starting to run for features:'%s' with %i threads" % (featureSet, threads)
print "Estimators: " , estimators
print "==============================================================="


# Get attributes and class for the images
x_data, targets_data = getAttributes(featureSet)
#x_data, targets_data = getAttributes('hara-img200')

# Separate in train & test
X_train, X_test, y_train, y_test = cross_validation.train_test_split(x_data, targets_data, test_size=0.2, random_state=0)

#Perform a grid search to find best method and parameters
print("We will start training with %i images. Each image with %i attributes." % (len(y_train), len(X_train[0])))
#raw_input('Press any key to continue:')
bestEstimators = gridSearch(X_train, y_train, estimators, featureSet, threads)

print("Training finished. Found %i estimators for featureSet '%s'." % (len(bestEstimators), featureSet))
#raw_input('Press any key to show the report for each estimator:')

# call the next script to load results
print("")
os.system("python savedEstimators.py")




	
