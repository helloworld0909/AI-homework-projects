from collections import defaultdict
import numpy as np


class Corpus(object):

    def __init__(self):
        self.id2word = {}
        self.word2id = {}
        self.docs = []
        self.V = 0
        self.M = 0

    def load_text(self, filepath):
        """        
        :param filepath: 
        Each line is a document which contains many words, and words are separated by whitespace
        
        """

        input_file = open(filepath, 'r')

        for line in input_file:
            doc = line.strip().split(' ')
            doc_id = np.empty(len(doc), dtype='intc')
            for index, word in enumerate(doc):
                if word not in self.word2id:
                    current_id = len(self.word2id)
                    self.word2id[word] = current_id
                    self.id2word[current_id] = word

                    doc_id[index] = current_id
                else:
                    doc_id[index] = self.word2id[word]
            self.docs.append(doc_id)

        self.V = len(self.word2id)
        self.M = len(self.docs)

        input_file.close()

    def load_ldac(self, filepath):

        input_file = open(filepath, 'r')

        for line in input_file:
            raw_doc = line.strip().split(' ')
            doc = []
            for wordcount in raw_doc[1:]:
                word, count = map(int, wordcount.split(':'))
                if word + 1 > self.V:
                    self.V = word + 1
                doc.extend([word] * count)
            self.docs.append(np.array(doc, dtype='intc'))

        self.M = len(self.docs)

        input_file.close()


if __name__ == '__main__':
    corpus = Corpus()
    corpus.load_ldac('input/reuters.ldac')
    print corpus.V


