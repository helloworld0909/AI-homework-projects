from collections import defaultdict
import numpy as np
import string
import json


class Corpus(object):

    def __init__(self):
        self.id2word = {}
        self.word2id = {}
        self.context = {}
        self.docs = []
        self.V = 0
        self.M = 0

    def load_text(self, filepath):
        """        
        :param filepath: The path of the input file
        Each line is a document which contains many words, and words are separated by whitespace.
        This method can load data and build vocabulary at the same time.
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

    def load_vocabulary(self, filepath):

        with open(filepath, 'r') as input_file:
            for index, line in enumerate(input_file):
                word = line.strip()
                self.word2id[word] = index
                self.id2word[index] = word

    def load_context(self, filepath):

        with open(filepath, 'r') as input_file:
            for index, line in enumerate(input_file):
                self.context[index] = line.strip()

    def load_movie(self, filepath):

        with open(filepath, 'r') as input_file:
            csv_first_line = input_file.readline()
            for index, line in enumerate(input_file):
                movieId = line.strip().split(',')[0]
                self.word2id[movieId] = index
                self.id2word[index] = movieId
            self.V = len(self.word2id)

    def load_rating(self, filepath, positive_threshold=4, atleast_rated = 0):

        rating_dict = defaultdict(list)
        with open(filepath, 'r') as input_file:
            csv_first_line = input_file.readline()

            for line in input_file:
                userId, movieID, rating, _ = line.strip().split(',')
                if float(rating) >= positive_threshold:
                    rating_dict[userId].append(self.word2id[movieID])

        for k, v in sorted(rating_dict.items(), key=lambda kv:int(kv[0])):
            if len(v) >= atleast_rated:
                self.docs.append(np.array(v, dtype='intc'))

        self.M = len(self.docs)

    def load_reuters(self, filepath, stopwordpath):

        preprocessor = Preprocessor(stopwordpath)
        keyplace = 'usa'
        self.context[keyplace] = []

        with open(filepath, 'r') as input_file:
            docs_json = json.load(input_file)
        for item_id, item in enumerate(docs_json):
            if 'body' not in item:
                continue
            doc = []
            body = item['body']
            for word in body.strip().split():
                word = preprocessor.remove_punctuation(word.strip())
                if not preprocessor.is_stopword(word):
                    if word not in self.word2id:
                        current_id = len(self.word2id)
                        self.word2id[word] = current_id
                        self.id2word[current_id] = word
                    doc.append(self.word2id[word])
            self.docs.append(np.array(doc, dtype='intc'))

            if "places" in item and keyplace in item["places"]:
                self.context[keyplace].append(1)
            else:
                self.context[keyplace].append(0)

        self.V = len(self.word2id)
        self.M = len(self.docs)

class Preprocessor(object):

    def __init__(self, filepath):
        self.stopword = set()
        self.load_stopword(filepath)
        self.punctuation = string.punctuation

    def load_stopword(self, filepath):
        with open(filepath, 'r') as input_file:
            for line in input_file:
                self.stopword.add(line.strip())

    def is_stopword(self, word):
        return word in self.stopword

    def remove_punctuation(self, word):
        for c in self.punctuation:
            word = word.replace(c, '')
        return word




if __name__ == '__main__':
    # corpus = Corpus()
    # corpus.load_ldac('input/reuters.ldac')
    # corpus.load_vocabulary('input/reuters.tokens')

    corpus = Corpus()
    corpus.load_movie('input/ml-latest-small/movies.csv')
    corpus.load_rating('input/ml-latest-small/ratings.csv')
    print corpus.docs
    print corpus.id2word[48]

