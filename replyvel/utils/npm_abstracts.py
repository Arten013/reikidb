from .edge import EdgeStore, Edge
from jstatutree.element import Element
import abc
from collections import Counter
import numpy as np
from queue import Queue
from typing import Sequence, Union
from pathlib import Path


class AbstractTraverser(abc.ABC):
    def __init__(self, query: Element, target: Element, edge_store: EdgeStore, threshold, *, logger=None):
        self.query = query
        self.target = target
        self.edge_queue = Queue()
        self.edge_store = edge_store
        for e in edge_store.iter_edges():
            self.edge_queue.put(e)

    def rise_query_parent(self, qnode: Element, stop_before_none=False) -> Union[None, Element]:
        if qnode is None:
            return None
        query_parent = self.get_query_parent(qnode)
        if query_parent is None:
            return qnode if stop_before_none else None
        elif len(list(query_parent.iterXsentence_code())) == len(list(qnode.iterXsentence_code())):
            return self.rise_query_parent(query_parent, stop_before_none)
        return qnode

    def rise_target_parent(self, tnode: Element, stop_before_none=False) -> Union[None, Element]:
        if tnode is None:
            return None
        target_parent = self.get_target_parent(tnode)
        if target_parent is None:
            return tnode if stop_before_none else None
        elif len(list(target_parent.iterXsentence_code())) == len(list(tnode.iterXsentence_code())):
            return self.rise_target_parent(target_parent, stop_before_none)
        return tnode

    def get_target_parent(self, tnode: Element) -> Union[None, Element]:
        return self.target.find_by_code(str(Path(tnode.code).parent))

    def get_query_parent(self, qnode: Element) -> Union[None, Element]:
        return self.query.find_by_code(str(Path(qnode.code).parent))

    @abc.abstractmethod
    def traverse(self):
        pass


class AbstractScorer(abc.ABC):  # todo: from other_result to calc_cache
    def __init__(self):
        self._cache = {"leaf_count": {}}

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

    def score_range(self, edge: Edge, edge_store: EdgeStore):
        return self.upper_limit(edge, edge_store), self.lower_limit(edge, edge_store)

    def upper_limit(self, edge: Edge, edge_store: EdgeStore):
        return 1.0

    def lower_limit(self, edge: Edge, edge_store: EdgeStore):
        return 0.0

    def calc_tleaf_count(self, edge):
        leaf_count = self._cache['leaf_count'].get(edge.tnode.code, None)
        if leaf_count is None:
            leaf_count = sum(1 for _ in edge.tnode.iterXsentence(include_code=False, include_value=False))
            self._cache['leaf_count'][edge.tnode.code] = leaf_count
        return leaf_count

    def calc_qleaf_count(self, edge):
        leaf_count = self._cache['leaf_count'].get(edge.qnode.code, None)
        if leaf_count is None:
            leaf_count = sum(1 for _ in edge.qnode.iterXsentence(include_code=False, include_value=False))
            self._cache['leaf_count'][edge.qnode.code] = leaf_count
        return leaf_count

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
