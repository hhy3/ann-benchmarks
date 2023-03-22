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

    def fit(self, X):
        if self.metric == "IP":
            X = preprocessing.normalize(X, "l2", axis=1)
        self.d = X.shape[1]
        self.d_align = (self.d + 63) // 64 * 64
        X = np.pad(X, pad_width=((0, 0), (0, self.d_align - self.d)))
        self.p = glass.Index(2, dim=len(
            X[0]), metric=self.metric, R=self.R, L=self.L)
        self.p.add(X)
        g = self.p.get_graph()
        g.save('fm.glass')
        self.searcher = glass.Searcher(g, X, self.metric, "SQ8")
        # self.searcher.warmup(8)

    def set_query_arguments(self, ef):
        print("FUCK")
        self.searcher.set_ef(ef)
        self.ef = ef

    def prepare_query(self, q, n):
        if self.metric == 'IP':
            q = q / np.linalg.norm(q)
        self.q = np.pad(q, pad_width=(0, self.d_align - self.d))
        self.n = n

    def run_prepared_query(self):
        self.res = self.searcher.search(
            self.q, self.n, min(self.ef, self.n * 2))

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
            self.queries, self.n, min(self.ef, self.n * 2))

    def get_batch_results(self):
        return self.result.reshape(self.nq, -1)

    def freeIndex(self):
        del self.p
        del self.searcher


class GlassFP16(BaseANN):
    def __init__(self, metric, method_param):
        self.metric = {'angular': 'IP', 'euclidean': 'L2'}[metric]
        self.R = method_param['R']
        self.L = method_param['L']
        self.name = 'glass_(%s)' % (method_param)

    def fit(self, X):
        if self.metric == "IP":
            X = preprocessing.normalize(X, "l2", axis=1)
        self.d = X.shape[0]
        self.d_align = (self.d + 63) // 64 * 64
        X = np.pad(X, pad_width=((0, 0), (0, self.d - self.d_align)))
        self.p = glass.Index(2, dim=len(
            X[0]), metric=self.metric, R=self.R, L=self.L)
        self.p.add(X)
        g = self.p.get_graph()
        self.searcher = glass.Searcher(
            g, X, self.metric, 'FP16')
        self.searcher.warmup()

    def set_query_arguments(self, ef):
        self.searcher.set_ef(ef)

    def query(self, v, n):
        return self.searcher.search(v, n)

    def freeIndex(self):
        del self.p


class GlassBF16(BaseANN):
    def __init__(self, metric, method_param):
        self.metric = {'angular': 'IP', 'euclidean': 'L2'}[metric]
        self.R = method_param['R']
        self.L = method_param['L']
        self.name = 'glass_(%s)' % (method_param)

    def fit(self, X):
        if self.metric == "IP":
            X = preprocessing.normalize(X, "l2", axis=1)
        self.p = glass.Index(2, dim=len(
            X[0]), metric=self.metric, R=self.R, L=self.L)
        self.p.add(X)
        self.g = self.p.get_graph()
        self.searcher = glass.Searcher(
            self.g, X, self.metric, 'BF16')
        self.searcher.warmup()

    def set_query_arguments(self, ef):
        self.searcher.set_ef(ef)
        self.ef = ef

    def query(self, v, n):
        return self.searcher.search(v, n)

    def freeIndex(self):
        del self.p


class GlassFP32(BaseANN):
    def __init__(self, metric, method_param):
        self.metric = {'angular': 'IP', 'euclidean': 'L2'}[metric]
        self.R = method_param['R']
        self.L = method_param['L']
        self.name = 'glass_(%s)' % (method_param)

    def fit(self, X):
        if self.metric == "IP":
            X = preprocessing.normalize(X, "l2", axis=1)
        self.d = X.shape[0]
        self.d_align = (self.d + 63) // 64 * 64
        X = np.pad(X, pad_width=((0, 0), (0, self.d - self.d_align)))
        self.p = glass.Index(2, dim=len(
            X[0]), metric=self.metric, R=self.R, L=self.L)
        self.p.add(X)
        self.g = self.p.get_graph()
        self.searcher = glass.Searcher(
            self.g, X, self.metric, 'FP32')
        self.searcher.warmup()

    def set_query_arguments(self, ef):
        self.searcher.set_ef(ef)

    def query(self, v, n):
        return self.searcher.search(v, n)

    def freeIndex(self):
        del self.p
