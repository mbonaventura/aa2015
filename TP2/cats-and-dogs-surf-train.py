import os
import sys
import json
import glob
import argparse
import mahotas as mh
from mahotas.features import surf
import numpy as np
from sklearn.cluster import MiniBatchKMeans
#
#
def clusterSurfFeatures(X_train_surf_features, n_clusters, all_instance_filenames):
	print 'Clustering', len(X_train_surf_features), 'features (k=' + str(n_clusters) + ')'
	estimator = MiniBatchKMeans(n_clusters=n_clusters)
	estimator.fit_transform(X_train_surf_features)
	#
	x_data = []
	instance_no = 0
	saved_features = {}
	for instance in surf_features:
		clusters = estimator.predict(instance)
		features = np.bincount(clusters)
		imagename = all_instance_filenames[instance_no]
		if len(features) < n_clusters:
			features = np.append(features, np.zeros((1, n_clusters-len(features))))
		#print features
		#
		imagename = os.path.basename(imagename)
		saved_features[imagename] = ';'.join(str(x) for x in features)
		instance_no += 1
	return saved_features
#
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
surf_features = []
#
image_counter = 0
for filename in all_instance_filenames:
	image_counter+=1
	imagename = os.path.basename(filename)	
	#
	print 'Processing image: ', image_counter, filename
	image = mh.imread(filename, as_grey=True)
	surf_data = surf.surf(image)[:, 5:]
	surf_features.append(surf_data)
#		

X_train_surf_features = np.concatenate(surf_features)

basefilename = args.featuresFileName

for k in (50, 100, 200, 300):
	saved_features = clusterSurfFeatures(X_train_surf_features, k, all_instance_filenames)	#n_clusters = 300
	json.dump(saved_features, open(basefilename+"-c"+str(k)+".json",'w'))
#
#
