import numpy as np
import logging
from corpus import Corpus


class LDA(object):

    def __init__(self, n_topic, alpha=0.1, beta=0.1, n_iter=1000):
        self.K = n_topic
        self.alpha = alpha
        self.beta = beta
        self.n_iter = n_iter

        self.logger = logging.getLogger('LDA')

        assert alpha > 0 and beta > 0, 'Alpha and beta should be larger than zero'
        assert isinstance(n_topic, int), 'n_topic should be an integer'


    def fit(self, corpus, algorithm='GS'):
        """
        
        :param corpus: 
        corpus.Corpus()
        
        :param algorithm: 
        'GS'    ->  Gibbs sampling
        'VI'    ->  Variational Inference
        
        :return: LDA
        """
        assert isinstance(corpus, Corpus), 'Input should be Corpus type'

        if algorithm == 'GS':
            self._fit_Gibbs(corpus)
        elif algorithm == 'VI':
            self._fit_inference(corpus)
        else:
            raise ValueError("algorithm must be either 'GS' or 'VI'")
        return self



    def _fit_Gibbs(self, corpus):
        pass

    def _initialize(self, corpus):
        pass

    def _fit_inference(self, corpus):
        pass
