import os
import sys
import json
import glob
import argparse
import numpy as np
import mahotas.features
from mahotas import lbp
from mahotas.features import surf
from sklearn import decomposition
from sklearn.cluster import MiniBatchKMeans
#
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

#---------------------------------------------------------------------------------------
# Main
#---------------------------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument("-t", "--trainingSetPath", help="Path to Training Set", required="True")
parser.add_argument("-fc", "--featuresColor",  help="Color Features File", default="")
parser.add_argument("-fg", "--featuresGray",   help="Color Features File", default="")
parser.add_argument("-fh", "--featuresHara",   help="Haraclick Features File", default="")
parser.add_argument("-fl", "--featuresLbp",    help="LBP Features File", default="")
parser.add_argument("-fs", "--featuresSurf",   help="SURF Features File", default="")
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
#
Y_data = []
X_data = []
X1_data = []
for img in imagename_list:
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
for img in imagename_list:
	this_texture_features = []
	# Features sobre 'texturas': histogramas de haraclick, lbp y surf
	this_texture_features.extend(train_hara_features[img]) if (args.featuresHara != "")	else ""
	this_texture_features.extend(train_lbp_features[img])  if (args.featuresLbp != "")	else ""
	this_texture_features.extend(train_surf_features[img]) if (args.featuresSurf != "")	else ""
	#
	X2_data.append(this_texture_features)

#X2_data = do_pca(X2_data)

for i in range(0, len(X1_data)):
	combined_features = []																																																																																	
	combined_features.extend(X1_data[i])
	combined_features.extend(X2_data[i])
	X_data.append(combined_features)

#

print "Training info loaded!", len(X_data), len(Y_data)

#--------------------------------------------------------------------------------------
# Cargo las imagenes a evaluar, calculo sus features (en base a las que cargue del modelo)
#
image_targets = {}
for f in glob.glob(args.trainingSetPath+'*.jpg'):
	target = 1 if 'cat' in f else 0		
	imagename = os.path.basename(f)
	image_targets[imagename] = target
#
#
image_counter = 0
surf_all_hist = {}
gray_features = {}
color_features = {}
hara_features = {}
lbp_features = {}
surf_features = {}

for imagename in image_targets:
	image_counter+=1
	filename = args.trainingSetPath + imagename
	print 'Processing image: ', image_counter, filename

	# Abro la imagen en escala de grises
	image = mahotas.imread(filename, as_grey=True)

	if (args.featuresGray != ""):	# Extraer histograma de grises
		gray_hist, bins = np.histogram(image.flatten(), 256, [0, 256])
		gray_features[imagename] = gray_hist
	#	
	if (args.featuresLbp != ""):	# Extraer lbp (sobre la misma imagen del anterior, en escala de grises)
		radius = 3
		points = 4 * radius							# Number of points to be considered as neighbourers
		lbp_hist = lbp.lbp(image, radius, points, ignore_zeros=False)
		lbp_features[imagename] = lbp_hist
	#
	if (args.featuresSurf != ""):	# Extraer surf (sobre la misma imagen del anterior, en escala de grises)
		surf_features = surf.surf(image)[:, 5:]
		surf_all_hist[imagename] = surf_features
	#			
	# Abro la imagen en colores
	image = mahotas.imread(filename, as_grey=False)
	#
	if (args.featuresColor != ""):	# Extraer histograma de colores
		color_hist, bins = np.histogram(image.flatten(), 256, [0, 256])
		color_features[imagename] = color_hist
	#
	if (args.featuresHara != ""):	# Extraer histograma haraclick (sobre la misma imagen del anterior, en colores)
		hara_hist = mahotas.features.haralick(image).mean(0)
		hara_features[imagename] = hara_hist

#------------------------------------------------------------------------------------------------------------------	
if (args.featuresSurf != ""):	# Clusterizo las features de 'surf'
	k = 300
	surf_features = clusterSurfFeatures(surf_all_hist, k)
#------------------------------------------------------------------------------------------------------------------
#
#------------------------------------------------------------------------------------------------------------------
# Ver para combinar: http://scikit-learn.org/stable/auto_examples/feature_stacker.html#example-feature-stacker-py
#
y_data = []
x_data = []
x1_data = []
for img in image_targets:
	target = 1 if 'cat' in img else 0		
	y_data.append(target)
	#
	this_color_features = []
	# Features sobre 'colores': histogramas de grises y colores 
	this_color_features.extend(train_color_features[img]) if (args.featuresColor != "")	else ""
	this_color_features.extend(train_gray_features[img])  if (args.featuresGray != "")	else ""
	x1_data.append(this_color_features)	
	#print img, len(train_lbp_features[img]), len(this_combined_features)
#
#x1_data = do_pca(x1_data)
#
x2_data = []
for img in image_targets:
	this_texture_features = []
	# Features sobre 'texturas': histogramas de haraclick, lbp y surf
	this_texture_features.extend(train_hara_features[img]) if (args.featuresHara != "")	else ""
	this_texture_features.extend(train_lbp_features[img])  if (args.featuresLbp != "")	else ""
	this_texture_features.extend(train_surf_features[img]) if (args.featuresSurf != "")	else ""
	#
	x2_data.append(this_texture_features)

#x2_data = do_pca(x2_data)

for i in range(0, len(x1_data)):
	combined_features = []																																																																																	
	combined_features.extend(x1_data[i])
	combined_features.extend(x2_data[i])
	x_data.append(combined_features)
#
#
print 'Training... '

#
#clf = RandomForestClassifier()
clf1 = RandomForestClassifier(max_depth=5, n_estimators=20, max_features=1)
#clf2 = SVC(kernel="linear", C=0.025)	#Kernel: linear, poly, rbf
#clf2 = GradientBoostingClassifier(n_estimators=100)
#clf3 = linear_model.LogisticRegression(C=1e5)
clf2 = RandomForestClassifier(random_state=1)
clf3 = RandomForestClassifier(random_state=2)

print 'Testing... '

eclf1 = VotingClassifier(estimators=[('rf', clf1), ('svm', clf2), ('gb', clf3)], voting='hard')
eclf1 = eclf1.fit(X_data, Y_data)

predictions = eclf1.predict(x_data)

#print y_data
#print predictions	
#clf.fit(X_train, y_train)
print classification_report(y_data, predictions)

#predictions = clf.predict(X_test)

#print y_test
#print predictions
#scores = cross_validation.cross_val_score(clf, x_data, targets_data, cv=5)
#print ""
#print "CV Scores....: ", scores
#print "CV Mean score: ", sum(scores)/(len(scores)*1.0)
