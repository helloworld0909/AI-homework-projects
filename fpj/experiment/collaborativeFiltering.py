from __future__ import absolute_import
import os
import sys

cwd = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(cwd)
sys.path.append(os.path.dirname(cwd))

from fpj.lda.corpus import Corpus
from fpj.lda.lda import LDA

menu_path = cwd + '/input/ml-20m/'

valid_split = 0.1
positive_threshold = 4.0
atleast_rated = 30
n_topic = 100

def load():
    corpus = Corpus()
    corpus.load_movie(menu_path + 'movies.csv')
    corpus.load_rating(menu_path + 'ratings_2m.csv', positive_threshold=positive_threshold, atleast_rated=atleast_rated)
    print corpus.M

    return corpus

def fit_movieLens():
    corpus = load()

    # Recommended choice of parameters
    alpha = 1.0 / n_topic
    beta = 200.0 / corpus.V

    model = LDA(n_topic=n_topic, alpha=alpha, beta=beta)

    model.fit(corpus, valid_split=valid_split, n_iter=100)

    model.save_model(protocol=2, filepath='save/')

def collaborative_filtering():
    corpus = load()

    model = LDA(n_topic=20)
    model.load_model(filepath='save/')

    X = corpus.docs[-int(valid_split * corpus.M):]

    Y = map(lambda d:d[-1], X)
    X = map(lambda d:d[:-1], X)

    generate_prob = model.predict(X)

    print model.predictive_perplexity(generate_prob, Y)


if __name__ == '__main__':
    fit_movieLens()
    collaborative_filtering()
