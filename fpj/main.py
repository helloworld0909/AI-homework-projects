from corpus import Corpus
from lda import LDA

menu_path = 'input/'

def fit_reuters():
    corpus = Corpus()
    corpus.load_ldac(menu_path + 'reuters.ldac')
    model = LDA(n_topic=20)
    model.fit(corpus, n_iter=50)

    model.save_model(protocol=2)

def output_reuters():
    model = LDA()
    model.load_model()

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
    model.fit(corpus, valid_split=0.1, n_iter=10)

    perplexity = model.perplexity(corpus.docs)
    print perplexity

if __name__ == '__main__':
    output_reuters()

