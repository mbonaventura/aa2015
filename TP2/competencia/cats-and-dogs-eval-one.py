import os
import sys
import json
import glob
import argparse
import numpy as np
#
import mahotas.features
from mahotas import lbp
from mahotas.features import surf
#
from sklearn import decomposition
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import *
from sklearn import cross_validation
from sklearn.svm import SVC
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier

#---------------------------------------------------------------------------------------
def norm_hist(hist):
	return [float(i)/sum(hist) for i in hist]
#---------------------------------------------------------------------------------------
def clusterSurfFeatures(surf_all_hist, n_clusters):
	#
	all_hists = []
	for imagename in surf_all_hist:		
		all_hists.append(surf_all_hist[imagename])
	#
	X_train_surf_features = np.concatenate(all_hists)
	#		
	print 'Clustering', len(X_train_surf_features), 'features (k=' + str(n_clusters) + ')'
	estimator = MiniBatchKMeans(n_clusters=n_clusters)
	estimator.fit_transform(X_train_surf_features)
	#	
	final_features = {}
	for imagename in surf_all_hist:
		instance = surf_all_hist[imagename]
		#
		clusters = estimator.predict(instance)
		features = np.bincount(clusters)
		#
		if len(features) < n_clusters:
			features = np.append(features, np.zeros((1, n_clusters-len(features))))
		#print features
		#		
		final_features[imagename] = features		
	return final_features
	
#---------------------------------------------------------------------------------------
def number_of_pc(variance_ratio, threshold):
	i = 0
	vr = variance_ratio[0]	
	while vr < threshold:
		i = 1
		vr+=variance_ratio[i]
	return (i+1)
#---------------------------------------------------------------------------------------
def do_pca(data):
	pca = decomposition.PCA(whiten=True)
	pca.fit(data)
	#print pca.explained_variance_ratio_
	nopc = number_of_pc(pca.explained_variance_ratio_, 0.99) + 12	
	#print pca.explained_variance_ratio_, nopc
	t_data = pca.transform(data)
	#print t_data
	np_data = np.asarray(t_data)
	X = np_data[:,0:nopc] 		# Retorno las primeras "nopc", que explican el 0.99 de la varianza.
	return X

#---------------------------------------------------------------------------------------
def json2FeaturesMap(data):
	tmp = {}	
	for img in data:
		tmp[img] = map(float, data[img].split(';'))
	return tmp, data.keys()
#
#---------------------------------------------------------------------------------------
# Main
#---------------------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("-fc", "--featuresColor",  help="Color Features File", default="")
parser.add_argument("-fg", "--featuresGray",   help="Color Features File", default="")
parser.add_argument("-fh", "--featuresHara",   help="Haraclick Features File", default="")
parser.add_argument("-fl", "--featuresLbp",    help="LBP Features File", default="")
parser.add_argument("-fs", "--featuresSurf",   help="SURF Features File", default="")
parser.add_argument("-label", "--labelModel",   help="Label para el modelo", default="")
args = parser.parse_args()

#--------------------------------------------------------------------------------------
# Cargo los parametros para el modelo
train_color_features = {}
train_gray_features = {}
train_hara_features = {}
train_lbp_features = {}
train_surf_features = {}
#
imagename_list = []
if (args.featuresColor != ""):		
	train_color_features, imagename_list = json2FeaturesMap(json.load(open(args.featuresColor)))
	print "[TRAIN] 'Color' features loaded!"
if (args.featuresGray != ""):
	train_gray_features, imagename_list = json2FeaturesMap(json.load(open(args.featuresGray)))
	print "[TRAIN] 'Gray' features loaded!"
if (args.featuresHara != ""):		
	train_hara_features, imagename_list = json2FeaturesMap(json.load(open(args.featuresHara)))
	print "[TRAIN] 'Haraclick' features loaded!"
if (args.featuresLbp != ""):		
	train_lbp_features, imagename_list = json2FeaturesMap(json.load(open(args.featuresLbp)))
	print "[TRAIN] 'LBP' features loaded!"
if (args.featuresSurf != ""):		
	train_surf_features, imagename_list = json2FeaturesMap(json.load(open(args.featuresSurf)))
	print "[TRAIN] 'SURF' features loaded!"
#



image_list = []
for img in imagename_list:
	image_list.append(img)

#
Y_data = []
X_data = []
X1_data = []
for img in image_list:
	target = 1 if 'cat' in img else 0		
	Y_data.append(target)
	#
	this_color_features = []
	# Features sobre 'colores': histogramas de grises y colores 
	this_color_features.extend(train_color_features[img]) if (args.featuresColor != "")	else ""
	this_color_features.extend(train_gray_features[img])  if (args.featuresGray != "")	else ""
	X1_data.append(this_color_features)	
	#print img, len(train_lbp_features[img]), len(this_combined_features)
#

#X1_data = do_pca(X1_data)

#
X2_data = []
for img in image_list:
	this_texture_features = []
	# Features sobre 'texturas': histogramas de haraclick, lbp y surf
	this_texture_features.extend(train_hara_features[img]) if (args.featuresHara != "")	else ""
	this_texture_features.extend(train_lbp_features[img])  if (args.featuresLbp != "")	else ""
	this_texture_features.extend(train_surf_features[img]) if (args.featuresSurf != "")	else ""
	#
	X2_data.append(this_texture_features)

#X2_data = do_pca(X2_data)

for i in range(0, len(image_list)):
	combined_features = []																																																																																	
	combined_features.extend(X1_data[i])
	combined_features.extend(X2_data[i])
	X_data.append(combined_features)

#
print "Training info loaded!"	#, len(X_data), len(Y_data)

#--------------------------------------------------------------------------------------
#
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X_data, Y_data, test_size=0.3, random_state=0)

print 'Training... '

# NOTE: Hasta ahora, con estos params x defecto la mejor performance (P=0.72) fue con:
#	clf1 = RF(5, 20, 1)
#	clf2 = GradientBoosting(100)
#	clf3 = Logistic(C=1e5)
#	y voting soft

#
#------------------------------------------------------------------------------------------------------------------
#clf1 = RandomForestClassifier(max_depth=5, n_estimators=20, max_features=1)
clf3 = GradientBoostingClassifier(n_estimators=100, loss='deviance', learning_rate=0.1, max_depth=10)	# Sacado de la tablita de Grid_Search
#clf3 = linear_model.LogisticRegression(C=0.0005, penalty='l2')						# Sacado de la tablita de Grid_Search
#------------------------------------------------------------------------------------------------------------------
#eclf1 = VotingClassifier(estimators=[('rf', clf1), ('svm', clf2), ('gb', clf3)], voting='soft')
#eclf1 = eclf1.fit(X_train, y_train)

clf3.fit(X_train, y_train)

print 'Testing... '
predictions = clf3.predict(X_test)

print classification_report(y_test, predictions)


# Guardo el modelo
from sklearn.externals import joblib
joblib.dump(clf3, "./models/"+args.labelModel)

#print y_data
#print predictions	
#clf.fit(X_train, y_train)
#predictions = clf.predict(X_test)
#print y_test
#print predictions
#scores = cross_validation.cross_val_score(clf, x_data, targets_data, cv=5)
#print ""
#print "CV Scores....: ", scores
#print "CV Mean score: ", sum(scores)/(len(scores)*1.0)
