from corpus import Corpus
from lda import LDA

menu_path = 'input/'

if __name__ == '__main__':

    corpus = Corpus()
    corpus.load_ldac(menu_path + 'reuters.ldac')
    model = LDA(n_topic=20)
    model.fit(corpus)
