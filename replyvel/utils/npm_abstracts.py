from .edge import EdgeStore, Edge
from jstatutree.element import Element
import abc
from collections import Counter
import numpy as np
from queue import Queue
from typing import Sequence


class AbstractTraverser(abc.ABC):
    def __init__(self, query: Element, target: Element, edge_store: EdgeStore, *, logger=None):
        self.query = query
        self.target = target
        self.edge_queue = Queue()
        self.edge_store = edge_store
        for e in edge_store.iter_edges():
            self.edge_queue.put(e)

    @abc.abstractmethod
    def traverse(self):
        pass


class AbstractScorer(abc.ABC): # todo: from other_result to calc_cache
    @abc.abstractmethod
    def scoring(self, edge:Edge, edge_store: EdgeStore, **other_results):
        pass

    @property
    def is_parallel(self):
        return False

    def reformatting_leaf_edges(self, *edges: Sequence[Edge]) -> Sequence[Edge]:
        return edges

class AbstractActivator(abc.ABC):
    def __init__(self):
        self.is_parallel = False
        self.activate = self.activate_core

    def serialize(self):
        self.is_parallel = False
        self.activate = self.activate_core

    def parallelize(self):
        self.is_parallel = True
        self.activate = np.vectorize(self.activate_core)

    def initial_edges(self, edges: list) -> EdgeStore: # todo: add scorer to argument
        edge_store = EdgeStore()
        for e in edges:
            q, t, score = e.to_tuple()
            edge_store.add_edge(q, t, self.activate(score))
        return edge_store

    @abc.abstractmethod
    def activate_core(self, val:float) -> float:
        pass
