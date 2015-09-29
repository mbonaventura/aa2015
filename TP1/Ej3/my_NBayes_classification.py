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


def get_minibatch(doc_iter, size, pos_class):
    """Extract a minibatch of examples, return a tuple X_text, y.
    Note: size is before excluding invalid docs with no topics assigned.
    """
    data = [(u'{title}\n\n{body}'.format(**doc), pos_class in doc['topics'])
            for doc in itertools.islice(doc_iter, size)
            if doc['topics']]
    if not len(data):
        return np.asarray([], dtype=int), np.asarray([], dtype=int)
    X_text, y = zip(*data)
    return X_text, np.asarray(y, dtype=int)

#------------------------------------------------------------------------------------------------
# Main
#
#------------------------------------------------------------------------------------------------

infile = sys.argv[1]
hfile = codecs.open(infile, encoding='utf-8')

vec = sys.argv[2]
#

swfile = ""
stop_words_list = []
try:
    swfile = sys.argv[3]
    with open(swfile, 'r') as sfile:
        for line in sfile.readlines():
            stop_words_list.append(line.strip())
    #
    print stop_words_list
except:
    pass

#
data = []
line =""
for line in hfile.readlines():
    line_array = line.strip().split(';')
    #
    line_dict = {}  
    line_dict['body'] = line_array[3]
    line_dict['topics'] = [line_array[2].strip()]
    line_dict['title'] = line_array[4]
    data.append(line_dict)
#

data_stream = data
n_total_documents = len(data)

# Create the vectorizer and limit the number of features to a reasonable maximum
# El valor original de n_features es 2^18. stop_words: string {english}, list, or None (default)
#
if vec == 'hash':
    print 'Inicializando hashing vectorizer...'
    vectorizer = HashingVectorizer(decode_error='ignore', n_features=2 ** 20, non_negative=True, stop_words=stop_words_list, binary=False)
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

names_classes = ['cultura', 'deportes', 'economia', 'espectaculos', 'internacionales', 'politica','seguridad','sociedad','tecnologia']

positive_class = 'tecnologia'  # deportes, seguridad



for positive_class in names_classes:
    # First we hold out a number of examples to estimate accuracy
    n_test_documents = 5000
    X_test_text, y_test = get_minibatch(data_stream, n_test_documents, positive_class   )
    #
    if vec == 'hash':
        X_test = vectorizer.transform(X_test_text)
    #
    if vec == 'tfidf':
        X_test = vectorizer.fit_transform(X_test_text)
    #
    if vec == 'count':
        X_test = vectorizer.fit_transform(X_test_text)
    #

    print("Test set for class (%s) is %d documents (%d positive)" % (positive_class, len(y_test), sum(y_test))) 

    n_remaining_documents = n_total_documents - n_test_documents
    X_train_text, y_train = get_minibatch(data_stream, n_remaining_documents, positive_class)
    X_train = vectorizer.transform(X_train_text)
    #

    cls = MultinomialNB(alpha=0.01) # alpha: Additive (Laplace/Lidstone) smoothing parameter (0 for no smoothing).

    y_pred = cls.fit(X_train, y_train).predict(X_test)
    #print X_test.shape, y_test.shape
    #predictions = cls.predict(X_test)
    #print 'Accuracy: ', cls.score(X_test, y_test)
    print 'Accuracy : ', accuracy_score(y_test, y_pred)
    print 'Precision: ', precision_score(y_test, y_pred)
    print 'Recall   : ', recall_score(y_test, y_pred)


print 'Matriz de confusion del ultimo :-)'
# Compute confusion matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Show confusion matrix in a separate window
plt.matshow(cm)
plt.title('Confusion matrix')
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()
#
    