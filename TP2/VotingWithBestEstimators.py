import sys
import numpy as np
from sklearn import cross_validation
from sklearn.metrics import classification_report
from sklearn.metrics import *
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from classifier_search import *
from attribute_extraction import *



# Get attributes and class for the images
featureSet = 'surf-c300-norm' # 'bp-r5' 'hara-img200'
x_data, targets_data = getAttributes(featureSet)

# Separate in train & test
#X_train, X_test, y_train, y_test = cross_validation.train_test_split(x_data, targets_data, test_size=0.2, random_state=0)
X_train = x_data
y_train = targets_data

# create the estimators with corresponding parameters
estimators = []
scores = []
estimators.append(('LogisticRegression', LogisticRegression(penalty='l2', C=0.005))) # expected 0.774
scores.append(0.774)
estimators.append(('SVC', SVC(kernel='rbf', C=100, gamma=0.001))) # expected 0.758
scores.append(0.758)
estimators.append(('GradientBoostingClassifier', GradientBoostingClassifier(n_estimators=150, loss='deviance', learning_rate=0.1, max_depth=3))) # expected 0.728
scores.append(0.728)
estimators.append(('BaggingClassifier', BaggingClassifier(max_features=0.5, max_samples=0.5, base_estimator=KNeighborsClassifier()))) # expected 0.718
scores.append(0.718)
estimators.append(('RandomForestClassifier', RandomForestClassifier(max_features=100, n_estimators=100, max_depth=100))) # expected 0.709
scores.append(0.709)

# Voting
voting = VotingClassifier(estimators=estimators, voting='hard');
print "Starting to train VotingClassifier"
voting.fit(X_train, y_train)
#voting_prediction = voting.predict(X_test)
#
#print("")
#print("Testing VOTING 'hard' with TEST data:")		    
#print("========================================================================================")
#print classification_report(y_test, voting_prediction)
#print 'Precision: ', precision_score(y_test, voting_prediction)
#print 'Recall: ', recall_score(y_test, voting_prediction)
#print 'Accuracy: ', accuracy_score(y_test, voting_prediction)
saveEstimator(voting, 1, './estimators/final_surf300_hard') 

voting = VotingClassifier(estimators=estimators, voting='soft', weights=scores);
voting.fit(X_train, y_train)
#voting_prediction = voting.predict(X_test)

#print("")
#print("Testing VOTING 'soft' with TEST data:")		    
#print("========================================================================================")
#print classification_report(y_test, voting_prediction)
#print 'Precision: ', precision_score(y_test, voting_prediction)
#print 'Recall: ', recall_score(y_test, voting_prediction)
#print 'Accuracy: ', accuracy_score(y_test, voting_prediction)
#saveEstimator(voting, precision_score(y_test, voting_prediction), './estimators/%s_votingSoft' % featureSet) 
saveEstimator(voting, 1, './estimators/final_surf300_soft') 
