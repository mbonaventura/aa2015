import os.path
import re, sys
import codecs
import itertools

import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import *
from sklearn.cross_validation import train_test_split

def setPositiveClass(topics, positive_class):
    num_class = []
    for top in topics:
        if (top == positive_class):
            num_class.append(1)
        else:
            num_class.append(0)
    return num_class 


#------------------------------------------------------------------------------------------------
# Main
#
#------------------------------------------------------------------------------------------------

infile = sys.argv[1]
hfile = codecs.open(infile, encoding='utf-8')

vec = sys.argv[2]
#
X_data = []
Y_data = []

for line in hfile.readlines():
    line_array = line.strip().split(';')
    this_text = line_array[3] + ' ' + line_array[4]
    X_data.append(this_text)
    this_class = line_array[2].strip()
    #if (this_class in ('politica', 'internacionales', 'seguridad', 'economia')):
    #    this_class = 'resto'
    #if (this_class in ('espectaculos', 'tecnologia', 'sociedad', 'cultura')):
    #    this_class = 'resto'
    Y_data.append(this_class)
#



#
vectorizer = ''
# Create the vectorizer and limit the number of features to a reasonable maximum
# El valor original de n_features es 2^18. stop_words: string {english}, list, or None (default)
#
if vec == 'hash':
    print 'Inicializando hashing vectorizer...'
    vectorizer = HashingVectorizer(decode_error='ignore', n_features=2 ** 14, non_negative=True, binary=False)
#
if vec == 'tfidf':
    print 'Inicializando tfidf vectorizer...'
    vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5)
#
if vec == 'count':
    print 'Inicializando count vectorizer...'
    vectorizer = CountVectorizer(max_df=0.5, binary=False)
#
#
class_names = ['cultura', 'deportes', 'economia', 'espectaculos', 'internacionales', 'politica', 'seguridad', 'sociedad','tecnologia']
#class_names = ['deportes', 'resto']

sum_precision = 0
sum_recall = 0
sum_f1 = 0
tcm = []
#
print 'Clase\t\tTrain +/t \tTest +/tot\tAcc\tPrec\tRec\tF1'
print '------------------------------------------------------------------------------'
#
for positive_class in class_names:
    
    X_train_text, X_test_text, y_train, y_test = train_test_split(X_data, Y_data, test_size=.3, random_state=0)

    #
    X_train = []
    X_test = []

    if vec == 'hash':
        X_train = vectorizer.transform(X_train_text)        
    #
    if vec == 'tfidf':
        X_train = vectorizer.fit_transform(X_train_text)
    #
    if vec == 'count':
        X_train = vectorizer.fit_transform(X_train_text)
    #
    X_test  = vectorizer.transform(X_test_text)
    #

    y_train = setPositiveClass(y_train, positive_class) 
    y_test  = setPositiveClass(y_test, positive_class)
    #

    #cls = MultinomialNB(alpha=0.01) # alpha: Additive (Laplace/Lidstone) smoothing parameter (0 for no smoothing).
    cls = MultinomialNB()
    y_pred = cls.fit(X_train, y_train).predict(X_test)
    #
    this_accuracy  = accuracy_score(y_test, y_pred)
    this_precision = precision_score(y_test, y_pred)
    this_recall    = recall_score(y_test, y_pred)
    this_f1        = f1_score(y_test, y_pred)
    
    # Compute confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    print '%15s\t%4d/%d\t%4d/%d\t%0.4f\t%0.4f\t%0.4f\t%0.4f' % (positive_class, sum(y_train), len(y_train), sum(y_test), len(y_test), this_accuracy, this_precision, this_recall, this_f1),
    print '\t',cm[0], cm[1]

    #
    if tcm==[]:
        tcm=cm
    else:
        tcm+=cm
    #
    sum_precision+=this_precision
    sum_recall+=this_recall
    sum_f1+=this_f1
    #
    
    # Plot de las curvas ROC
    probas_ = cls.fit(X_train, y_train).predict_proba(X_test)
    # Compute ROC curve and area the curve
    fpr, tpr, thresholds = roc_curve(y_test, probas_[:, 1])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=1, label=positive_class + ' (area = %0.2f)' % (roc_auc))



#
#
print '------------------------------------------------------------------------------'
print ''
print 'Resumen'
print tcm
print ''
print 'AVG Precision: %0.4f'% (sum_precision/(len(class_names)+0.0))
print 'AVG Recall   : %0.4f'% (sum_recall/(len(class_names)+0.0)) 
print 'AVG F1       : %0.4f'% (sum_f1/(len(class_names)+0.0))
#
#
plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='')
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()


# Show confusion matrix in a separate window
#plt.matshow(cm)
#plt.title('Confusion matrix')
#plt.colorbar()
#plt.ylabel('True label')
#plt.xlabel('Predicted label')
#plt.show()
#
    