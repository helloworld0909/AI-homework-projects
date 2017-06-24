import logging
import os
import pickle

import numpy as np

from corpus import Corpus
from util import weightedRandomChoice

logging.basicConfig(
    level=logging.DEBUG,
    format="[%(asctime)s] %(levelname)s: %(message)s"
)


class LDA(object):
    def __init__(self, n_topic=10, alpha=0.1, beta=0.1):
        self.V = 0
        self.K = n_topic
        self.alpha = alpha
        self.beta = beta
        self.valid_split = 0.0

        assert alpha > 0 and beta > 0, 'Alpha and beta should be larger than zero'
        assert isinstance(n_topic, int), 'n_topic should be an integer'

        self.logger = logging.getLogger('LDA')

    def fit(self, corpus, valid_split=0.0, algorithm='GS', n_iter=1000, verbose=True):
        """

        :param corpus:
        corpus.Corpus()

        :param valid_split:

        :param n_iter:

        :param algorithm:
        'GS'    ->  Gibbs sampling
        'VI'    ->  Variational Inference

        :param verbose:
        True: print log information

        :return: LDA
        """
        assert isinstance(corpus, Corpus), 'Input should be Corpus type'

        self.valid_split = valid_split
        V = self.V = corpus.V
        M = int(corpus.M * (1 - valid_split))
        if algorithm == 'GS':
            self._fit_GS(corpus.docs[: M], V, n_iter, verbose)
        elif algorithm == 'VI':
            pass
        else:
            raise ValueError("algorithm must be either 'GS' or 'VI'")
        return self

    def _fit_GS(self, docs, V, n_iter, verbose=True):
        M = len(docs)
        self._initialize(docs, V, M)
        for it in range(n_iter):
            update_k_count = 0

            for m, doc in enumerate(docs):
                for n, word in enumerate(doc):
                    old_k = self.z_mn[m][n]
                    self.n_mk[m][old_k] -= 1
                    self.n_kt[old_k][word] -= 1
                    self.n_m[m] -= 1
                    self.n_k[old_k] -= 1

                    new_k = self._sample_topic(m, word)

                    self.n_mk[m][new_k] += 1
                    self.n_kt[new_k][word] += 1
                    self.n_m[m] += 1
                    self.n_k[new_k] += 1

                    self.z_mn[m][n] = new_k

                    if new_k != old_k:
                        update_k_count += 1

            self._read_out_parameters()
            if verbose:
                self.logger.info('<iter{0}> perplexity: {1:.6g} update rate: {2:.6g}'.format(it, self.perplexity(docs),
                                                                                  float(update_k_count) / self.N_sum))

    def _initialize(self, docs, V, M):
        self.N_sum = 0
        self.n_mk = np.zeros((M, self.K), dtype='intc')
        self.n_m = np.zeros(M, dtype='intc')
        self.n_kt = np.zeros((self.K, V), dtype='intc')
        self.n_k = np.zeros(self.K, dtype='intc')

        self.phi = np.empty((self.K, V), dtype='float64')
        self.theta = np.empty((M, self.K), dtype='float64')

        self.z_mn = []
        for m, doc in enumerate(docs):
            self.N_sum += len(doc)
            z_m = np.empty(len(doc), dtype='intc')
            for n, word in enumerate(doc):
                init_k = int(np.random.random(1) * self.K)
                z_m[n] = init_k
                self.n_mk[m][init_k] += 1
                self.n_kt[init_k][word] += 1
                self.n_m[m] += 1
                self.n_k[init_k] += 1
            self.z_mn.append(z_m)

    def _sample_topic(self, m, word):

        prob_k = self._full_conditional(m, word)
        prob_k /= prob_k.sum()

        new_k = weightedRandomChoice(prob_k)
        return new_k

    def _full_conditional(self, m, word):
        """
        Compute p(z_i = k|z_-i, w)
        :param m: m-th document
        :param word: The id of the word
        :return: p(z_i = k|z_-i, w)
        """
        return (self.n_kt[:, word] + self.beta) / (self.n_k + self.V * self.beta) * (self.n_mk[m, :] + self.alpha)

    def _read_out_parameters(self):
        for k in range(self.K):
            self.phi[k] = (self.n_kt[k] + self.beta) / (self.n_k[k] + self.V * self.beta)

        for m in range(self.n_m.size):
            self.theta[m] = (self.n_mk[m] + self.alpha) / (self.n_m[m] + self.K * self.alpha)

    def _fit_inference(self, corpus, valid_split, n_iter):
        pass

    def topic_word(self, n_top_word=10, corpus=None):
        if not hasattr(self, 'phi'):
            raise Exception('You should fit model first')
        else:
            topic_word_list = []
            for k in range(self.K):
                word_list = []
                for index in self.phi[k].argsort()[-n_top_word:]:
                    if corpus is not None:
                        word_list.append(corpus.id2word[index])
                    else:
                        word_list.append(index)
                topic_word_list.append(word_list)
            return topic_word_list

    def document_topic(self, n_top_topic=1, corpus=None, limit=10):
        if not hasattr(self, 'theta'):
            raise Exception('You should fit model first')
        else:
            M = self.theta[:, 0].size
            document_topic_list = []
            for m in range(min(limit, M)):
                topic_list = []
                for index in self.theta[m].argsort()[-n_top_topic:]:
                    topic_list.append(index)

                if corpus is not None:
                    document_topic_list.append((corpus.context[m], tuple(topic_list)))
            return document_topic_list

    def save_model(self, filepath='model/', protocol=0):
        if not os.path.exists(filepath):
            os.mkdir(filepath)
        with open(filepath + 'model.pkl', 'wb') as output_file:
            pickle.dump(self.K, output_file, protocol)
            pickle.dump(self.alpha, output_file, protocol)
            pickle.dump(self.beta, output_file, protocol)
            pickle.dump(self.valid_split, output_file, protocol)
            pickle.dump(self.V, output_file, protocol)
            pickle.dump(self.N_sum, output_file, protocol)
            pickle.dump(self.z_mn, output_file, protocol)
            pickle.dump(self.phi, output_file, protocol)
            pickle.dump(self.theta, output_file, protocol)

    def load_model(self, filepath='model/'):
        with open(filepath + 'model.pkl', 'rb') as input_file:
            self.K = pickle.load(input_file)
            self.alpha = pickle.load(input_file)
            self.beta = pickle.load(input_file)
            self.valid_split = pickle.load(input_file)
            self.V = pickle.load(input_file)
            self.N_sum = pickle.load(input_file)
            self.z_mn = pickle.load(input_file)
            self.phi = pickle.load(input_file)
            self.theta = pickle.load(input_file)

    def perplexity(self, docs):
        if not hasattr(self, 'theta'):
            raise Exception('You should fit model first')

        M = self.theta.shape[0]
        return self._perplexity(docs[:M], self.phi, self.theta, self.N_sum)

    def _perplexity(self, docs, phi, theta, N_sum):
        expindex = 0.0
        for m, doc in enumerate(docs):
            for word in doc:
                p = np.dot(theta[m, :], phi[:, word])
                expindex += np.log(p)
        return np.exp(-expindex / N_sum)
