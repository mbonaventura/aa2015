import os
import sys
import json
import glob
import argparse
import mahotas as mh
from mahotas.features import surf
#import cv2
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
	image_surf_data = []
	if  imagename in saved_features:
  		image_surf_data = map(float, saved_features[imagename].split(';'))
  	else:
		image = mh.imread(filename, as_grey=True)
		image_surf_data = surf.surf(image)[:, 5:]
    	doSave = 1
	image_features.append(image_surf_data)
	#
	all_surf_data = ""
	for img_sa in image_surf_data:
		list2str_sa = ';'.join(str(x) for x in img_sa)
		all_surf_data += list2str_sa + ":"	
	#
	#print all_surf_data[:-1]	
	saved_features[imagename] = all_surf_data[:-1]
	

if (doSave == 1):
	json.dump(saved_features, open(args.featuresFileName,'w'))

print "Features saved to: " + args.featuresFileName
