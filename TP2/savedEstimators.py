import sys
import numpy as np
from sklearn import cross_validation
from sklearn.metrics import classification_report
from sklearn.metrics import *
from sklearn.ensemble import VotingClassifier
from classifier_search import *
from attribute_extraction import *

featureSet = 'surf-c100' # 'bp-r5' 'hara-img200'

# Load estimators
print("Loading estimators from './estimators/%s'" % featureSet)
bestEstimators = loadEstimators('./estimators/%s' % featureSet)
estimators = [("%s" % estimator, estimator) for (estimator, score) in bestEstimators]
scores = [score for (estimator, score) in bestEstimators]

## TODO: no esta bueno tener que reentranar a voting. Grabarlo antes
# Get attributes and class for the images
x_data, targets_data = getAttributes(featureSet)

# Separate in train & test
X_train, X_test, y_train, y_test = cross_validation.train_test_split(x_data, targets_data, test_size=0.2, random_state=0)

# print the result for the best estimators
bestbest_estimator, bestbest_score = bestEstimators[0]
bestbest_score = 0
for estimator, predictedScore in bestEstimators:
	predictions = estimator.predict(X_test)

	print("")
	print("Testing %s with TEST data:" % estimator)		    
	print("========================================================================================")
	print classification_report(y_test, predictions)
	print 'Precision: ', precision_score(y_test, predictions)
	print 'Recall: ', recall_score(y_test, predictions)
	print 'Accuracy: ', accuracy_score(y_test, predictions)
	print 'Predicted score: ', predictedScore

	if(bestbest_score < precision_score(y_test, predictions)):
		bestbest_score = precision_score(y_test, predictions)
		bestbest_estimator = estimator

# Voting
voting = VotingClassifier(estimators=estimators, voting='hard');
## TODO: no esta bueno tener que reentranar a voting. Grabarlo antes
voting.fit(X_train, y_train)
voting_prediction = voting.predict(X_test)

print("")
print("Testing VOTING 'hard' with TEST data:")		    
print("========================================================================================")
print classification_report(y_test, voting_prediction)
print 'Precision: ', precision_score(y_test, voting_prediction)
print 'Recall: ', recall_score(y_test, voting_prediction)
print 'Accuracy: ', accuracy_score(y_test, voting_prediction)
saveEstimator(voting, precision_score(y_test, voting_prediction), './estimators/%s_votingHard' % featureSet) 

scores = [score for (estimator, score) in bestEstimators]
voting = VotingClassifier(estimators=estimators, voting='soft', weights=scores);
voting.fit(X_train, y_train)
voting_prediction = voting.predict(X_test)

print("")
print("Testing VOTING 'soft' with TEST data:")		    
print("========================================================================================")
print classification_report(y_test, voting_prediction)
print 'Precision: ', precision_score(y_test, voting_prediction)
print 'Recall: ', recall_score(y_test, voting_prediction)
print 'Accuracy: ', accuracy_score(y_test, voting_prediction)
saveEstimator(voting, precision_score(y_test, voting_prediction), './estimators/%s_votingSoft' % featureSet) 

print("========================================================================================")
print("========================================================================================")
print("The very best best estimator is (or de Votings?): (Precision %f )" % bestbest_score)
print(bestbest_estimator)