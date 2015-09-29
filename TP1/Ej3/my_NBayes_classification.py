from glob import glob
import itertools
import os.path
import re, sys
import tarfile
import time
import codecs

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

from sklearn.externals.six.moves import html_parser
from sklearn.externals.six.moves import urllib
from sklearn.datasets import get_data_home
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.linear_model import Perceptron
from sklearn.naive_bayes import MultinomialNB



#------------------------------------------------------------------------------------------------
# Main
#
#------------------------------------------------------------------------------------------------

infile = sys.argv[1]
hfile = codecs.open(infile, encoding='utf-8')
#
data = []
line =""
for line in hfile.readlines():
    line_array = line.strip().split(';')
    #
    line_dict = {}
    #print (line_array)
    line_dict['body'] = line_array[3]
    line_dict['topics'] = [line_array[2].strip()]
    line_dict['title'] = line_array[4]
    #print (line_dict['body'])
    #print (line_dict['topics'])
    #print (line_dict['title'])
    data.append(line_dict)
    #print (line_array)
#

data_stream = data
n_total_documents = len(data)


# Create the vectorizer and limit the number of features to a reasonable maximum
# El valor original de n_features es 2^18
vectorizer = HashingVectorizer(decode_error='ignore', n_features=2 ** 10, non_negative=True)
#

all_classes = np.array([0, 1])

positive_class = 'deportes'  # deportes, seguridad


def get_minibatch(doc_iter, size, pos_class=positive_class):
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


# Here are some classifiers that support the `partial_fit` method
partial_fit_classifiers = {
    'NB Multinomial': MultinomialNB(alpha=0.01),
}


# First we hold out a number of examples to estimate accuracy
n_test_documents = 5000
X_test_text, y_test = get_minibatch(data_stream, n_test_documents)
X_test = vectorizer.transform(X_test_text)

print("Test set is %d documents (%d positive)" % (len(y_test), sum(y_test)))

#get_minibatch(data_stream, n_test_documents)

# We will feed the classifier with mini-batches of 1000 documents; this means
# we have at most 1000 docs in memory at any time.  The smaller the document
# batch, the bigger the relative overhead of the partial fit methods.

n_remaining_documents = n_total_documents - n_test_documents

X_train_text, y_train = get_minibatch(data_stream, n_remaining_documents)

X_train = vectorizer.transform(X_train_text)

for cls_name, cls in partial_fit_classifiers.items():
    # update estimator with examples in the current mini-batch
    #print X_train.shape, y_train.shape
    #
    cls.partial_fit(X_train, y_train, classes=all_classes)
    #
    #print X_test.shape, y_test.shape
    print cls.score(X_test, y_test)
    #
    