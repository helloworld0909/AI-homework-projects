from __future__ import absolute_import
import os
import sys

cwd = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(cwd)
sys.path.append(os.path.dirname(cwd))

from fpj.lda.corpus import Corpus
from fpj.lda.lda import LDA

menu_path = cwd + '/input/reuters/'

def fit_reuters(n_topic=20, n_iter=200, save=True):
    corpus = Corpus()
    corpus.load_ldac(menu_path + 'reuters.ldac')
    model = LDA(n_topic=n_topic)
    model.fit(corpus, n_iter=n_iter)
    if save:
        model.save_model(filepath='save/reuters_model.pkl', protocol=2)
    return model

def output_reuters():
    model = LDA()
    model.load_model(filepath='save/reuters_model.pkl')

    corpus = Corpus()
    corpus.load_ldac(menu_path + 'reuters.ldac')
    corpus.load_vocabulary(menu_path + 'reuters.tokens')
    corpus.load_context(menu_path + 'reuters.titles')

    topic_word = model.topic_word(n_top_word=10, corpus=corpus)
    print '\n'.join(map(str, topic_word))

    document_topic = model.document_topic(n_top_topic=1, corpus=corpus, limit=10)
    print '\n'.join(map(str, document_topic))

def main():
    corpus = Corpus()
    corpus.load_ldac(menu_path + 'reuters.ldac')
    model = LDA(n_topic=20)
    model.fit(corpus, valid_split=0.0, n_iter=10)

    perplexity = model.perplexity(corpus.docs)
    print perplexity

if __name__ == '__main__':
    fit_reuters()
    output_reuters()

