from __future__ import absolute_import
import os
import glass
import numpy as np
from ann_benchmarks.constants import INDEX_DIR
from ann_benchmarks.algorithms.base import BaseANN
from sklearn import preprocessing

class Glass(BaseANN):
    def __init__(self, metric, method_param):
        self.metric = {'angular': 'IP', 'euclidean': 'L2'}[metric]
        self.R = method_param['R']
        self.L = method_param['L']
        self.name = 'glass_(%s)' % (method_param)
        self.dir = 'indices'
        self.path = f'{self.metric}_R_{self.R}_L_{self.L}.glass'

    def fit(self, X):
        if self.metric == "IP":
            X = preprocessing.normalize(X, "l2", axis=1)
        self.d = X.shape[1]
        self.d_align = (self.d + 63) // 64 * 64
        X = np.pad(X, pad_width=((0, 0), (0, self.d_align - self.d)))
        if not os.path.exists(self.dir):
            os.mkdir(self.dir)
        if self.path in os.listdir(self.dir):
            g = glass.Graph()
            g.load(os.path.join(self.dir, self.path))
        else:
            p = glass.Index(2, dim=self.d_align,
                            metric=self.metric, R=self.R, L=self.L)
            p.add(X)
            g = p.get_graph()
            g.save(os.path.join(self.dir, self.path))
        self.searcher = glass.Searcher(g, X, self.metric, "SQ8")

    def set_query_arguments(self, ef):
        self.searcher.set_ef(ef)
        self.ef = ef

    def prepare_query(self, q, n):
        if self.metric == 'IP':
            q = q / np.linalg.norm(q)
        self.q = np.pad(q, pad_width=(0, self.d_align - self.d))
        self.n = n

    def run_prepared_query(self):
        self.res = self.searcher.search(
            self.q, self.n)

    def get_prepared_query_results(self):
        return self.res

    def prepare_batch_query(self, X, n):
        if self.metric == 'angular':
            X = preprocessing.normalize(X, axis=1, norm='l2')
        self.queries = np.pad(X, pad_width=(
            (0, 0), (0, self.d_align - self.d)))
        self.n = n
        self.nq = len(X)

    def run_batch_query(self):
        self.result = self.searcher.batch_search(
            self.queries, self.n, self.ef)

    def get_batch_results(self):
        return self.result.reshape(self.nq, -1)

    def freeIndex(self):
        del self.searcher
        
        
class GlassFP16(BaseANN):
    def __init__(self, metric, method_param):
        self.metric = {'angular': 'IP', 'euclidean': 'L2'}[metric]
        self.R = method_param['R']
        self.L = method_param['L']
        self.name = 'glass_(%s)' % (method_param)
        self.dir = 'indices'
        self.path = f'{self.metric}_R_{self.R}_L_{self.L}_fp16.glass'

    def fit(self, X):
        if self.metric == "IP":
            X = preprocessing.normalize(X, "l2", axis=1)
        self.d = X.shape[1]
        self.d_align = (self.d + 63) // 64 * 64
        X = np.pad(X, pad_width=((0, 0), (0, self.d_align - self.d)))
        if not os.path.exists(self.dir):
            os.mkdir(self.dir)
        if self.path in os.listdir(self.dir):
            g = glass.Graph()
            g.load(os.path.join(self.dir, self.path))
        else:
            p = glass.Index(2, dim=self.d_align,
                            metric=self.metric, R=self.R, L=self.L)
            p.add(X)
            g = p.get_graph()
            g.save(os.path.join(self.dir, self.path))
        self.searcher = glass.Searcher(g, X, self.metric, "FP16")

    def set_query_arguments(self, ef):
        self.searcher.set_ef(ef)
        self.ef = ef

    def prepare_query(self, q, n):
        if self.metric == 'IP':
            q = q / np.linalg.norm(q)
        self.q = np.pad(q, pad_width=(0, self.d_align - self.d))
        self.n = n

    def run_prepared_query(self):
        self.res = self.searcher.search(
            self.q, self.n)

    def get_prepared_query_results(self):
        return self.res

    def prepare_batch_query(self, X, n):
        if self.metric == 'angular':
            X = preprocessing.normalize(X, axis=1, norm='l2')
        self.queries = np.pad(X, pad_width=(
            (0, 0), (0, self.d_align - self.d)))
        self.n = n
        self.nq = len(X)

    def run_batch_query(self):
        self.result = self.searcher.batch_search(
            self.queries, self.n, self.ef)

    def get_batch_results(self):
        return self.result.reshape(self.nq, -1)

    def freeIndex(self):
        del self.searcher