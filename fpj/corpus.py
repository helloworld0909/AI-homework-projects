from collections import defaultdict
import numpy as np


class Corpus(object):

    def __init__(self):
        self.id2word = {}
        self.word2id = {}
        self.docs = None
        self.V = 0
        self.W = 0

    def load_text(self, filepath):
        """        
        :param filepath: 
        Each line is a document which contains many words, and words are separated by whitespace
        
        """

        input_file = open(filepath, 'r')

        docs = []
        for line in input_file:
            doc = line.strip().split(' ')
            doc_id = []
            for word in doc:
                if word not in self.word2id:
                    current_id = len(self.word2id)
                    self.word2id[word] = current_id
                    self.id2word[current_id] = word

                    doc_id.append(current_id)
                else:
                    doc_id.append(self.word2id[word])
            docs.append(doc_id)

        self.docs = np.array(docs)
        self.V = len(self.word2id)
        self.W = len(docs)

if __name__ == '__main__':
    corpus = Corpus()
    corpus.load_text('input/test.txt')
    print corpus.word2id
    print corpus.id2word
    print corpus.docs
    print corpus.V
    print corpus.W

