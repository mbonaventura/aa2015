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
args = parser.parse_args()

threads = int(args.threads)

for featureSet in files.keys() :
	print("")
	try:
		os.system("python cat-dogs-gridSearch.py -t %i -e few -f %s" % (threads, featureSet))
	except:
		print "Oops!  there was an error processing feature:'%s'/n" % (featureSet)
		continue

	print("")
	print("")





	
