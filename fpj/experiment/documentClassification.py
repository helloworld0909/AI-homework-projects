from __future__ import absolute_import
import os
import sys

cwd = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(cwd)
sys.path.append(os.path.dirname(cwd))

from fpj.lda.corpus import Corpus
from fpj.lda.lda import LDA

from fpj.experiment.documentModeling import fit_reuters




