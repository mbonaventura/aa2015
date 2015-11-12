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
def clusterSurfFeatures(definitive_surf_features, n_clusters, all_instance_filenames):
	X_train_surf_features = np.concatenate(definitive_surf_features)
	print 'Clustering', len(X_train_surf_features), 'features (k=' + str(n_clusters) + ')'
	#
	estimator = MiniBatchKMeans(n_clusters=n_clusters)
	estimator.fit_transform(X_train_surf_features)
	#
	x_data = []
	instance_no = 0
	saved_features = {}
	for instance in definitive_surf_features:
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
parser.add_argument("-f", "--featuresFileName", help="Output Features File", required="True")
args = parser.parse_args()
#
#
print "Loading from file..."
saved_surf_features = json.load(open(args.featuresFileName))
#
all_instance_filenames = []
definitive_surf_features = []
for f in saved_surf_features:
	all_instance_filenames.append(f)
	this_instance_surf_data = []
	surf_lists = saved_surf_features[f].split(":")
	for sl in surf_lists:
		this_data = map(float, sl.split(';'))
		this_instance_surf_data.append(this_data)
		#print this_data
	definitive_surf_features.append(this_instance_surf_data)	
#
#
basefilename = args.featuresFileName
for k in (50, 100, 200, 300):
	saved_features = clusterSurfFeatures(definitive_surf_features, k, all_instance_filenames)	#n_clusters = 300
	json.dump(saved_features, open(basefilename+"-c"+str(k)+".json",'w'))
#
#
