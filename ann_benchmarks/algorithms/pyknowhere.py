from __future__ import absolute_import
import os
import pyknowhere
import numpy as np
from ann_benchmarks.constants import INDEX_DIR
from ann_benchmarks.algorithms.base import BaseANN


class PyKnowhere(BaseANN):
    def __init__(self, metric, method_param):
        self.metric = {'angular': 'cosine', 'euclidean': 'l2'}[metric]
        self.params = method_param
        self.name = 'pyknowhere (%s)' % (self.params)

    def fit(self, X):
        self.p = pyknowhere.Index(self.metric, len(X[0]), len(X), self.params["M"], self.params["efConstruction"])
        data_labels = np.arange(len(X))
        self.p.add(np.asarray(X), data_labels)

    def set_query_arguments(self, ef):
        self.p.set_param(ef)

    def query(self, v, n):
        return self.p.search(v, k=n)

    def freeIndex(self):
        del self.p
