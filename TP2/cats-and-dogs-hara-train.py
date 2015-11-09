import os
import sys
import json
import glob
import argparse
import cv2
import mahotas.features

#from skimage.feature import local_binary_pattern
#from scipy.stats import itemfreq
#from sklearn.preprocessing import normalize
#
parser = argparse.ArgumentParser()
parser.add_argument("-t", "--trainingSetPath", help="Path to Training Set", required="True")
parser.add_argument("-f", "--featuresFileName", help="Output Features File", required="True")
args = parser.parse_args()

#
all_instance_filenames = []
all_instance_targets = []
for f in glob.glob(args.trainingSetPath+'*.jpg'):
	target = 1 if 'cat' in f else 0
	all_instance_filenames.append(f)
	all_instance_targets.append(target)
	#print f, target
#
saved_features = {}

if os.path.isfile(args.featuresFileName): 
	saved_features = json.load(open(args.featuresFileName))
#
#
image_features = []
image_counter = 0
doSave = 0
for filename in all_instance_filenames:
	image_counter+=1
	imagename = os.path.basename(filename)
	print 'Processing image: ', image_counter, filename
	#
	hist = []
	if  imagename in saved_features:
  		hist = map(float, saved_features[imagename].split(';'))
  	else:
		image = mahotas.imread(filename, as_grey=False)
		features = mahotas.features.haralick(image).mean(0)
    	doSave = 1
    #
	saved_features[imagename] = ';'.join(str(x) for x in features)

if (doSave == 1):
	json.dump(saved_features, open(args.featuresFileName,'w'))
	print "Features saved to: " + args.featuresFileName
