from __future__ import absolute_import
import sys
sys.path.append("install/lib-faiss")  # noqa
import numpy
from sklearn import preprocessing
import ctypes
import faiss
import numpy as np
from time import time

from ann_benchmarks.algorithms.base import BaseANN


class FaissFS(BaseANN):
    def __init__(self, metric, n_list):
        self._n_list = n_list
        self._metric = metric

    def fit(self, X):
        if X.dtype != numpy.float32:
            X = X.astype(numpy.float32)
        if self._metric == 'angular':
            faiss.normalize_L2(X)

        d = X.shape[1]
        faiss_metric = faiss.METRIC_INNER_PRODUCT if self._metric == 'angular' else faiss.METRIC_L2
        if faiss_metric == faiss.METRIC_INNER_PRODUCT:
            factory_string = f"IVF{self._n_list}_HNSW32,PQ{d//2}x4fs"
        else:
            factory_string = f"IVF{self._n_list},PQ{d//2}x4fsr"
        index = faiss.index_factory(d, factory_string, faiss_metric)
        index.train(X)
        index.add(X)
        refine = faiss.index_factory(d, "SQfp16")
        refine.add(X)
        index_refine = faiss.IndexRefine(index, refine)
        self.base_index = index
        self.refine_index = index_refine
        self.refine_index.k_factor = 10
        self.index = self.refine_index

    def set_query_arguments(self, n_probe):
        faiss.cvar.indexIVF_stats.reset()
        self._n_probe = n_probe
        self.base_index.nprobe = self._n_probe

    def prepare_query(self, q, n):
        if self._metric == 'angular':
            q /= numpy.linalg.norm(q)
        self.q = q
        self.n = n

    def run_prepared_query(self):
        _, I = self.index.search(np.expand_dims(self.q, axis=0), self.n)
        self.res = I[0]

    def get_prepared_query_results(self):
        return self.res
      
    def prepare_batch_query(self, X, n):
        if self._metric == 'angular':
            X = preprocessing.normalize(X, axis=1, norm='l2')
        self.n = n
        self.queries = X

    def run_batch_query(self):
        t = time()
        _, I = self.index.search(
            self.queries, self.n)
        print(time() - t)
        self.res = I

    def get_batch_results(self):
        return self.res

    def get_additional(self):
        return {"dist_comps": faiss.cvar.indexIVF_stats.ndis +      # noqa
                faiss.cvar.indexIVF_stats.nq * self._n_list}

    def __str__(self):
        return 'FaissIVFPQfs(n_list=%d, n_probe=%d)' % (self._n_list,
                                                                      self._n_probe)
