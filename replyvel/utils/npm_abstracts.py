from .edge import EdgeStore, Edge
from jstatutree.element import Element
import abc
from collections import Counter
import numpy as np
from queue import Queue
from typing import Sequence, Union
from pathlib import Path


class AbstractTraverser(abc.ABC):
    def __init__(self, query: Element, target: Element, edge_store: EdgeStore, *, logger=None):
        self.query = query
        self.target = target
        self.edge_queue = Queue()
        self.edge_store = edge_store
        for e in edge_store.iter_edges():
            self.edge_queue.put(e)

    def get_target_parent(self, tnode: Element) -> Union[None, Element]:
        target_parent = self.target.find_by_code(str(Path(tnode.code).parent))
        if target_parent is not None and len(target_parent) == len(tnode):
            return self.get_target_parent(target_parent)
        return target_parent

    def get_query_parent(self, qnode: Element) -> Union[None, Element]:
        query_parent = self.query.find_by_code(str(Path(qnode.code).parent))
        if query_parent is not None and len(query_parent) == len(qnode):
            return self.get_query_parent(query_parent)
        return query_parent

    @abc.abstractmethod
    def traverse(self):
        pass


class AbstractScorer(abc.ABC):  # todo: from other_result to calc_cache
    @abc.abstractmethod
    def scoring(self, edge: Edge, edge_store: EdgeStore, **other_results):
        pass

    @property
    def tags(self):
        return ['default']

    @property
    def is_parallel(self):
        return False

    def reformatting_leaf_edges(self, *edges: Edge) -> Sequence[Edge]:
        return edges


class AbstractActivator(abc.ABC):
    def __init__(self):
        self.is_parallel = False
        self.activate = self.activate_core

    def serialize(self) -> None:
        self.is_parallel = False
        self.activate = self.activate_core

    def parallelize(self) -> None:
        self.is_parallel = True
        self.activate = np.vectorize(self.activate_core)

    def initial_edges(self, edges: Sequence[Edge]) -> EdgeStore:  # todo: add scorer to argument
        edge_store = EdgeStore()
        for e in edges:
            q, t, score = e.to_tuple()
            edge_store.add(q, t, self.activate(score))
        return edge_store

    @abc.abstractmethod
    def activate_core(self, val: float) -> float:
        pass
