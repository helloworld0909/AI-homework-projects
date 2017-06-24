import sys
sys.path.append("..")

from lda.corpus import Corpus
from lda.lda import LDA

menu_path = '../input/ml-latest-small/'

def fit_movieLens():
    corpus = Corpus()
    corpus.load_movie(menu_path + 'movies.csv')
    corpus.load_rating(menu_path + 'ratings.csv', positive_threshold=3.5)

    model = LDA(n_topic=20)
    model.fit(corpus, n_iter=100)

    model.save_model(protocol=2, filepath='example/')

if __name__ == '__main__':
    fit_movieLens()
