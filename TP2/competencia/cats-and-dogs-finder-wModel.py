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
#
from sklearn.externals import joblib
#
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
		features = []
		try:
			clusters = estimator.predict(instance)
			features = np.bincount(clusters)
		except:
			features.append(0)
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
parser.add_argument("-t", "--testSetPath", help="Path to Test Set", required="True")
parser.add_argument("--fc", action='store_true', help="Use Color Features File", default=False)
parser.add_argument("--fg", action='store_true', help="Use Color Features File", default=False)
parser.add_argument("--fh", action='store_true', help="Use Haraclick Features File", default=False)
parser.add_argument("--fl", action='store_true', help="Use LBP Features File", default=False)
parser.add_argument("--fs", action='store_true', help="Use SURF Features File", default=False)
parser.add_argument("-m", "--model", help="Model to use", required="True")
parser.add_argument("-o", "--outFile", help="Output file", required="True")
args = parser.parse_args()
#--------------------------------------------------------------------------------------
#
# Cargo el modelo ya entrenado!
#
#
eclf1 = joblib.load('./models/'+args.model) 
print "Model loaded..."

#--------------------------------------------------------------------------------------
# Cargo las imagenes a evaluar, calculo sus features (en base a las que cargue del modelo)
#
image_targets = {}
for f in glob.glob(args.testSetPath+'*.jpg'):
	target = 1 if 'cat' in f else 0		
	imagename = os.path.basename(f)
	image_targets[imagename] = target
#


print "Loading", len(image_targets), "images..."
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
	filename = args.testSetPath + imagename
	print 'Processing image: ', image_counter, filename

	# Abro la imagen en escala de grises
	image = mahotas.imread(filename, as_grey=True)

	if (args.fc):	# Extraer histograma de grises
		gray_hist, bins = np.histogram(image.flatten(), 256, [0, 256])
		gray_features[imagename] = gray_hist
	#	
	if (args.fl):	# Extraer lbp (sobre la misma imagen del anterior, en escala de grises)
		radius = 3
		points = 4 * radius	 # Number of points to be considered as neighbourers
		lbp_hist = lbp.lbp(image, radius, points, ignore_zeros=False)
		lbp_features[imagename] = lbp_hist
	#
	if (args.fs):	# Extraer surf (sobre la misma imagen del anterior, en escala de grises)
		surf_features = surf.surf(image)[:, 5:]
		surf_all_hist[imagename] = surf_features
	#			
	# Abro la imagen en colores
	image = mahotas.imread(filename, as_grey=False)
	#
	if (args.fc):	# Extraer histograma de colores
		color_hist, bins = np.histogram(image.flatten(), 256, [0, 256])
		color_features[imagename] = color_hist
	#
	if (args.fh):	# Extraer histograma haraclick (sobre la misma imagen del anterior, en colores)
		hara_hist = mahotas.features.haralick(image).mean(0)
		hara_features[imagename] = hara_hist

#------------------------------------------------------------------------------------------------------------------	
if (args.fs):	# Clusterizo las features de 'surf'
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


image_list = []
for img in image_targets:
	image_list.append(img)
#
for img in image_list:
	target = 1 if 'cat' in img else 0		
	y_data.append(target)
	#
	this_color_features = []
	# Features sobre 'colores': histogramas de grises y colores 
	this_color_features.extend(color_features[img]) if (args.fc)	else ""
	this_color_features.extend(gray_features[img])  if (args.fg)	else ""
	x1_data.append(this_color_features)	
	#print img, len(train_lbp_features[img]), len(this_combined_features)
#
#x1_data = do_pca(x1_data)
#
x2_data = []
for img in image_list:
	this_texture_features = []
	# Features sobre 'texturas': histogramas de haraclick, lbp y surf
	this_texture_features.extend(hara_features[img]) if (args.fh)	else ""
	this_texture_features.extend(lbp_features[img])  if (args.fl)	else ""
	this_texture_features.extend(surf_features[img]) if (args.fs)	else ""
	#
	x2_data.append(this_texture_features)

#x2_data = do_pca(x2_data)

for i in range(0, len(image_list)):
	combined_features = []																																																																																	
	combined_features.extend(x1_data[i])
	combined_features.extend(x2_data[i])
	x_data.append(combined_features)
#
#

print 'Testing... '

predictions = eclf1.predict(x_data)
print classification_report(y_data, predictions)

print 'Saving results to:', args.outFile
fout = open(args.outFile, 'w')
for i in range(0, len(image_list)):
	fout.write(image_list[i]+','+str(predictions[i])+'\n')
fout.close()
#
