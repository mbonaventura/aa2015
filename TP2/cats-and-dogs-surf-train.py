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
	array = []
	with open(args.featuresFileName, "r") as fin:
		for line in fin:
			data = line.split("\t")
			k = data[0]
			saved_features[k] = data[1].strip()
		#
	#
#
image_counter = 0

fout = open(args.featuresFileName,'w')

for filename in all_instance_filenames:
	image_counter+=1
	imagename = os.path.basename(filename)	
	#
	image_surf_data = []
	if  imagename in saved_features:
		print 'Image already processed :-) ', image_counter, filename
		fout.write(imagename + "\t" + saved_features[imagename] + "\n")
  	else:
		print 'Processing image: ', image_counter, filename
		image = mh.imread(filename, as_grey=True)
		surf_data = surf.surf(image)[:, 5:]
		doSave = 1
		#
		str_surf_data = ""
		for img_sa in surf_data:
			#print img_sa 
			tmp_surf_data = ';'.join(str(x) for x in img_sa)
			str_surf_data = str_surf_data + tmp_surf_data + ":"
		#
		fout.write(imagename + "\t" + str_surf_data + "\n")
	#
#		
fout.close()
print "Features saved to: " + args.featuresFileName
#
