from corpus import Corpus
from lda import LDA

menu_path = 'input/'

def fit_reuters():
    corpus = Corpus()
    corpus.load_ldac(menu_path + 'reuters.ldac')
    model = LDA(n_topic=20, n_iter=1000)
    model.fit(corpus)

    model.save_model(protocol=2)

def output_reuters():
    model = LDA()
    model.load_model()

    corpus = Corpus()
    corpus.load_vocabulary(menu_path + 'reuters.tokens')
    corpus.load_context(menu_path + 'reuters.titles')

    topic_word = model.topic_word(n_top_word=10, corpus=corpus)
    print topic_word

    document_topic = model.document_topic(n_top_topic=1, corpus=corpus, limit=10)
    print document_topic

if __name__ == '__main__':
    output_reuters()

