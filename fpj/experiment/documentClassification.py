from __future__ import absolute_import
import os
import sys

cwd = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(cwd)
sys.path.append(os.path.dirname(cwd))

import numpy as np
from sklearn.linear_model import LogisticRegression

from fpj.lda.corpus import Corpus
from fpj.lda.lda import LDA

menu_path = cwd + '/input/reuters-21578/'

n_topic=20
n_iter=100

def fit(corpus):

    alpha = 1.0 / n_topic
    beta = 200.0 / corpus.V

    model = LDA(n_topic=n_topic, alpha=alpha, beta=beta)
    model.fit(corpus, n_iter=n_iter)
    model.save_model(filepath='save/reuters-21578/', protocol=2)

def linear_classifier_bench(corpus):
    from sklearn.preprocessing import OneHotEncoder
    X = []
    for doc in corpus.docs:
        doc_onehot = np.array()

def linear_classifier(lda_model, corpus):


    split_index = int(corpus.M * 0.9)
    X = lda_model.theta
    Y = np.array(corpus.context['usa'], dtype='int')

    X_train = X[:split_index]
    Y_train = Y[:split_index]

    X_test = X[split_index:]
    Y_test = Y[split_index:]

    lr = LogisticRegression()
    lr.fit(X_train, Y_train)

    Y_predict = lr.predict(X_test)

    precision = sum(map(lambda y1y2: int(y1y2[0] == y1y2[1]), zip(Y_test, Y_predict))) / float(len(X_test))

    print 'USA or Not USA test Precision = ', precision

if __name__ == '__main__':
    corpus = Corpus()
    corpus.load_reuters(menu_path + 'reuters.json', menu_path + 'stopword.txt')

    model = LDA()
    model.load_model(filepath='save/reuters-21578/')

    linear_classifier(model, corpus)





