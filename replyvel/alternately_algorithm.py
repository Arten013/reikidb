from jstatutree.element import Element
from scipy.stats import moment
from jstatutree import etypes
from pathlib import Path
from typing import Generator, Mapping, Union, Sequence
import math
import numpy as np
from collections import OrderedDict
from .utils.npm_abstracts import AbstractScorer, AbstractActivator, AbstractTraverser
from .utils.edge import Edge, EdgeStore, ParallelizedEdge, ParallelizedEdgeStore
from .utils.logger import get_logger
from queue import PriorityQueue
import re

import heapq
import datetime


def hinit(items) -> list:
    ret = [(0, e) for e in items]
    heapq.heapify(ret)
    return ret


def hpush(queue, item, priority):
    heapq.heappush(queue, (priority, item))


def hpop(queue):
    return heapq.heappop(queue)[1]


class PriorityEdgeQueue(object):
    def __init__(self):
        self.q = []

    def is_empty(self):
        return len(self.q) == 0

    def get(self) -> Edge:
        return heapq.heappop(self.q)[2]

    def put(self, item: Edge, priority=0):
        heapq.heappush(self.q, (priority, datetime.datetime.now(), item))


class EdgeQueue(object):
    def __init__(self):
        self.items = [None] * 1000000
        self.head = 0
        self.tail = 0

    def is_empty(self):
        return self.head == self.tail

    def get(self) -> Edge:
        if self.is_empty():
            raise Exception('Queue is empty.')
        ret: Edge = self.items[self.tail]
        self.tail += 1
        return ret

    def put(self, item: Edge):
        self.items[self.head] = item
        self.head += 1


class Traverser(AbstractTraverser):
    def traverse(self) -> Generator:
        logger = get_logger('Alternately.Traverser.traverse')
        logger.debug('Initial queue size: %d', self.edge_queue.qsize())
        while not self.edge_queue.empty():
            edge = self.edge_queue.get()
            if not edge.is_leaf_pair():
                if edge in self.edge_store:
                    logger.debug('skip: already scored')
                    # logger.info(str(self.edge_store.find_edge(edge.to_tuple())))
                    continue
                logger.debug('yield edge %s', str(edge))
                score = yield edge  # todo: implement halt(score, edges)
            parent_qnode = self.get_query_parent(edge.qnode)
            parent_tnode = self.get_target_parent(edge.tnode)
            if parent_qnode:
                self.edge_queue.put(Edge(parent_qnode, edge.tnode))
            if parent_tnode:
                self.edge_queue.put(Edge(edge.qnode, parent_tnode))
            if parent_qnode and parent_tnode:
                # self.edge_queue.put(Edge(parent_qnode, parent_tnode, 0))
                pass
        yield None


class AllTraverser(AbstractTraverser):  # todo: make
    def __init__(self, query: Element, target: Element, edge_store: EdgeStore, threshold, *, logger=None):
        self.query = query
        self.target = target
        self.threshold = threshold
        self.edge_queue = EdgeQueue()  # hinit(edge_store.iter_edges())
        self.edge_store = edge_store
        # print(self.match_lca)
        for e in edge_store.iter_edges():
            self.edge_queue.put(e)

    def traverse(self) -> Generator:
        logger = get_logger('Alternately.Traverser.traverse')
        while not self.edge_queue.is_empty():
            edge = self.edge_queue.get()
            if edge.is_leaf_pair():
                parent_qnode = self.rise_query_parent(edge.qnode, stop_before_none=True)
                parent_tnode = self.rise_target_parent(edge.tnode, stop_before_none=True)
                edge = Edge(parent_qnode, parent_tnode)
                if edge.is_leaf_pair():
                    item = yield False  # todo: implement halt(score, edges)
                else:
                    item = yield edge
                    if item is None:
                        continue
            elif edge in self.edge_store:  # or self.edge_store.has_ancestor_pair(edge, include_self=True, filter_mark_tag='delete'):
                # logger.info(str(self.edge_store.find_edge(edge.to_tuple())))
                continue
            else:
                item = yield edge
                if item is None:
                    continue
            score, scorer, edge_store, missing_link = item
            check = False
            parent_qnode = self.get_query_parent(edge.qnode)
            parent_tnode = self.get_target_parent(edge.tnode)
            raised_parent_qnode = self.rise_query_parent(self.get_query_parent(edge.qnode))
            raised_parent_tnode = self.rise_target_parent(self.get_target_parent(edge.tnode))
            if raised_parent_qnode is not None:
                parent_qnode = raised_parent_qnode
            if raised_parent_tnode is not None:
                parent_tnode = raised_parent_tnode
            future_edges = []
            if parent_qnode:
                future_edges.append(Edge(parent_qnode, edge.tnode))
            if parent_tnode:
                future_edges.append(Edge(edge.qnode, parent_tnode))
            for e in future_edges:
                self.edge_queue.put(e)
        yield None


class LCAPriorityTraverser(AbstractTraverser):  # todo: make
    def __init__(self, query: Element, target: Element, edge_store: EdgeStore, threshold, *, logger=None):
        self.query = query
        self.target = target
        self.threshold = threshold
        self.edge_queue = EdgeQueue()  # hinit(edge_store.iter_edges())
        self.edge_store = edge_store
        self.match_lca = self.edge_store.lca_pair_codes()

    def get_future_edges(self, edge):
        parent_qnode = self.get_query_parent(edge.qnode)
        parent_tnode = self.get_target_parent(edge.tnode)
        raised_parent_qnode = self.rise_query_parent(self.get_query_parent(edge.qnode))
        raised_parent_tnode = self.rise_target_parent(self.get_target_parent(edge.tnode))
        if raised_parent_qnode is not None:
            parent_qnode = raised_parent_qnode
        if raised_parent_tnode is not None:
            parent_tnode = raised_parent_tnode
        future_edges = []
        if parent_qnode:
            future_edges.append(Edge(parent_qnode, edge.tnode))
        if parent_tnode:
            future_edges.append(Edge(edge.qnode, parent_tnode))
        return future_edges

    def traverse(self):
        if not (len(self.match_lca[0]) == 21 and len(self.match_lca[1]) == 21):
            self.edge_queue.put(Edge(self.query.find_by_code(self.match_lca[0]), self.target.find_by_code(self.match_lca[1])))
            yielded_flag = False
            while not self.edge_queue.is_empty():
                edge = self.edge_queue.get()
                if edge.is_leaf_pair():
                    parent_qnode = self.rise_query_parent(edge.qnode, stop_before_none=True)
                    parent_tnode = self.rise_target_parent(edge.tnode, stop_before_none=True)
                    edge = Edge(parent_qnode, parent_tnode)
                    if edge.is_leaf_pair():
                        item = yield False
                    else:
                        item = yield edge
                        # if item is None:
                        #     continue
                elif edge in self.edge_store:
                    continue
                else:
                    item = yield edge
                    if item is None:
                        continue
                score, scorer, edge_store, _ = item
                if score < self.threshold:
                    continue
                yielded_flag = True
                for e in self.get_future_edges(edge):
                    self.edge_queue.put(e)
            if yielded_flag:
                pass
                #print('lca_skip!')
                #yield None
                #return
        for e in self.edge_store.iter_edges():
            self.edge_queue.put(e)
        while not self.edge_queue.is_empty():
            edge = self.edge_queue.get()
            in_lca_upper_bound = self.match_lca is not None \
                                 and edge.qnode.code in self.match_lca[0] \
                                 and edge.tnode.code in self.match_lca[1]
            if in_lca_upper_bound:
                continue
                pass
            if edge.is_leaf_pair():
                parent_qnode = self.rise_query_parent(edge.qnode, stop_before_none=True)
                parent_tnode = self.rise_target_parent(edge.tnode, stop_before_none=True)
                edge = Edge(parent_qnode, parent_tnode)
                if edge.is_leaf_pair():
                    item = yield False  # todo: implement halt(score, edges)
                else:
                    item = yield edge
                    if item is None:
                        continue
            elif edge in self.edge_store:
                continue
            elif self.edge_store.has_ancestor_pair(edge, include_self=True, filter_mark_tag='delete'):
                item = yield False
            else:
                item = yield edge
                if item is None:
                    continue
            score, scorer, edge_store, _ = item
            for e in self.get_future_edges(edge):
                self.edge_queue.put(e)
        yield None


class PriorityTraverser(AbstractTraverser):  # todo: make
    def __init__(self, query: Element, target: Element, edge_store: EdgeStore, threshold, *, logger=None):
        self.query = query
        self.target = target
        self.threshold = threshold
        self.edge_queue = hinit(edge_store.iter_edges())
        self.edge_store = edge_store

    def traverse(self) -> Generator:
        logger = get_logger('Alternately.Traverser.traverse')
        while len(self.edge_queue) > 0:
            priority, edge = heapq.heappop(self.edge_queue)
            # if "06/062014/0483" in edge.tnode.code:
            # print("prio:", priority, str(edge))
            if edge.is_leaf_pair():
                parent_qnode = self.rise_query_parent(edge.qnode, stop_before_none=True)
                parent_tnode = self.rise_target_parent(edge.tnode, stop_before_none=True)
                # print('before: ', edge)
                edge = Edge(parent_qnode, parent_tnode)
                if edge.is_leaf_pair():
                    item = yield False  # todo: implement halt(score, edges)
                else:
                    item = yield edge
                    if item is None:
                        continue
                # print('after: ', edge)
            elif edge in self.edge_store or self.edge_store.has_ancestor_pair(edge, include_self=True,
                                                                              filter_mark_tag='delete'):
                # logger.info(str(self.edge_store.find_edge(edge.to_tuple())))
                continue
            # print("prio:", priority, str(edge))
            else:
                # if "06/062014/0483" in edge.tnode.code:
                # print("yield:", priority, str(edge))
                item = yield edge
                if item is None:
                    continue
            score, scorer, edge_store, missing_link = item
            check = False
            if missing_link == 0 and score < self.threshold:
                # check = True
                continue
            parent_qnode = self.get_query_parent(edge.qnode)
            parent_tnode = self.get_target_parent(edge.tnode)
            raised_parent_qnode = self.rise_query_parent(self.get_query_parent(edge.qnode))
            raised_parent_tnode = self.rise_target_parent(self.get_target_parent(edge.tnode))
            if raised_parent_qnode is not None:
                parent_qnode = raised_parent_qnode
            if raised_parent_tnode is not None:
                parent_tnode = raised_parent_tnode
            future_edges = []
            if parent_qnode:
                future_edges.append(Edge(parent_qnode, edge.tnode))
            if parent_tnode:
                future_edges.append(Edge(edge.qnode, parent_tnode))
            if parent_qnode and parent_tnode:
                future_edges.append(Edge(parent_qnode, parent_tnode))
            # halt_flag = True
            # print(missing_link, edge)
            if missing_link > 0:
                # if True:
                for e in future_edges:
                    upper_limit, lower_limit = scorer.score_range(e, edge_store)
                    hpush(self.edge_queue, e, -lower_limit)

            else:
                for e in future_edges:
                    upper_limit, lower_limit = scorer.score_range(e, edge_store)
                    assert lower_limit <= scorer.scoring(e, edge_store)[
                        0] <= upper_limit, '{0:03}, {1:03}, {2:03}'.format(lower_limit,
                                                                           scorer.scoring(e, edge_store)[0],
                                                                           upper_limit)
                    if upper_limit >= self.threshold:
                        # if "06/062014/0483" in e.tnode.code:
                        # print("rise:", e)
                        hpush(self.edge_queue, e, -lower_limit)
                        # if check and scorer.scoring(e, edge_store)[0] >= self.threshold:
                        #     print('pruning error:')
                        #     e.score = scorer.scoring(e, edge_store)[0]
                        #     print(edge)
                        #     print(e)
                        #     raise Exception()
                        halt_flag = False
                # if halt_flag and len(future_edges) > 0:
                #     print("HALT:", str(edge))
                #     pass
        yield None


class DFSPriorityTraverser(AbstractTraverser):  # todo: make
    def __init__(self, query: Element, target: Element, edge_store: EdgeStore, threshold, *, logger=None):
        self.query = query
        self.target = target
        self.threshold = threshold
        self.edge_queue = hinit(edge_store.iter_edges())
        self.edge_store = edge_store

    def traverse(self) -> Generator:
        logger = get_logger('Alternately.Traverser.traverse')
        while len(self.edge_queue) > 0:
            priority, edge = heapq.heappop(self.edge_queue)
            # if "06/062014/0483" in edge.tnode.code:
            # print("prio:", priority, str(edge))
            if edge.is_leaf_pair():
                parent_qnode = self.rise_query_parent(edge.qnode, stop_before_none=True)
                parent_tnode = self.rise_target_parent(edge.tnode, stop_before_none=True)
                # print('before: ', edge)
                edge = Edge(parent_qnode, parent_tnode)
                if edge.is_leaf_pair():
                    item = yield False  # todo: implement halt(score, edges)
                else:
                    item = yield edge
                    if item is None:
                        continue
                # print('after: ', edge)
            elif edge in self.edge_store or self.edge_store.has_ancestor_pair(edge, include_self=True,
                                                                              filter_mark_tag='delete'):
                # logger.info(str(self.edge_store.find_edge(edge.to_tuple())))
                continue
            # print("prio:", priority, str(edge))
            else:
                # if "06/062014/0483" in edge.tnode.code:
                # print("yield:", priority, str(edge))
                item = yield edge
                if item is None:
                    continue
            score, scorer, edge_store, missing_link = item
            check = False
            if missing_link == 0 and score < self.threshold:
                # check = True
                continue
            parent_qnode = self.get_query_parent(edge.qnode)
            parent_tnode = self.get_target_parent(edge.tnode)
            raised_parent_qnode = self.rise_query_parent(self.get_query_parent(edge.qnode))
            raised_parent_tnode = self.rise_target_parent(self.get_target_parent(edge.tnode))
            if raised_parent_qnode is not None:
                parent_qnode = raised_parent_qnode
            if raised_parent_tnode is not None:
                parent_tnode = raised_parent_tnode
            future_edges = []
            if parent_qnode:
                future_edges.append(Edge(parent_qnode, edge.tnode))
            if parent_tnode:
                future_edges.append(Edge(edge.qnode, parent_tnode))
            if parent_qnode and parent_tnode:
                future_edges.append(Edge(parent_qnode, parent_tnode))
            for e in future_edges:
                upper_limit, lower_limit = scorer.score_range(e, edge_store)
                assert lower_limit <= scorer.scoring(e, edge_store)[0] <= upper_limit, '{0:03}, {1:03}, {2:03}'.format(
                    lower_limit, scorer.scoring(e, edge_store)[0], upper_limit)
                if upper_limit >= self.threshold:
                    # if "06/062014/0483" in e.tnode.code:
                    # print("rise:", e)
                    hpush(self.edge_queue, e, e.qnode.CATEGORY + e.tnode.CATEGORY)
                    # if check and scorer.scoring(e, edge_store)[0] >= self.threshold:
                    #     print('pruning error:')
                    #     e.score = scorer.scoring(e, edge_store)[0]
                    #     print(edge)
                    #     print(e)
                    #     raise Exception()
                    halt_flag = False
            # if halt_flag and len(future_edges) > 0:
            #     print("HALT:", str(edge))
            #     pass
        yield None


class BFSPriorityTraverser(AbstractTraverser):  # todo: make
    def __init__(self, query: Element, target: Element, edge_store: EdgeStore, threshold, *, logger=None):
        self.query = query
        self.target = target
        self.threshold = threshold
        self.edge_queue = hinit(edge_store.iter_edges())
        self.edge_store = edge_store

    def traverse(self) -> Generator:
        logger = get_logger('Alternately.Traverser.traverse')
        while len(self.edge_queue) > 0:
            priority, edge = heapq.heappop(self.edge_queue)
            # if "06/062014/0483" in edge.tnode.code:
            # print("prio:", priority, str(edge))
            if edge.is_leaf_pair():
                parent_qnode = self.rise_query_parent(edge.qnode, stop_before_none=True)
                parent_tnode = self.rise_target_parent(edge.tnode, stop_before_none=True)
                # print('before: ', edge)
                edge = Edge(parent_qnode, parent_tnode)
                if edge.is_leaf_pair():
                    item = yield False  # todo: implement halt(score, edges)
                else:
                    item = yield edge
                    if item is None:
                        continue
                # print('after: ', edge)
            elif edge in self.edge_store or self.edge_store.has_ancestor_pair(edge, include_self=True,
                                                                              filter_mark_tag='delete'):
                # logger.info(str(self.edge_store.find_edge(edge.to_tuple())))
                continue
            # print("prio:", priority, str(edge))
            else:
                # if "06/062014/0483" in edge.tnode.code:
                # print("yield:", priority, str(edge))
                item = yield edge
                if item is None:
                    continue
            score, scorer, edge_store, missing_link = item
            check = False
            if missing_link == 0 and score < self.threshold:
                # check = True
                continue
            parent_qnode = self.get_query_parent(edge.qnode)
            parent_tnode = self.get_target_parent(edge.tnode)
            raised_parent_qnode = self.rise_query_parent(self.get_query_parent(edge.qnode))
            raised_parent_tnode = self.rise_target_parent(self.get_target_parent(edge.tnode))
            if raised_parent_qnode is not None:
                parent_qnode = raised_parent_qnode
            if raised_parent_tnode is not None:
                parent_tnode = raised_parent_tnode
            future_edges = []
            if parent_qnode:
                future_edges.append(Edge(parent_qnode, edge.tnode))
            if parent_tnode:
                future_edges.append(Edge(edge.qnode, parent_tnode))
            if parent_qnode and parent_tnode:
                future_edges.append(Edge(parent_qnode, parent_tnode))
            for e in future_edges:
                upper_limit, lower_limit = scorer.score_range(e, edge_store)
                assert lower_limit <= scorer.scoring(e, edge_store)[0] <= upper_limit, '{0:03}, {1:03}, {2:03}'.format(
                    lower_limit, scorer.scoring(e, edge_store)[0], upper_limit)
                if upper_limit >= self.threshold:
                    # if "06/062014/0483" in e.tnode.code:
                    # print("rise:", e)
                    hpush(self.edge_queue, e, -e.qnode.CATEGORY - e.tnode.CATEGORY)
                    # if check and scorer.scoring(e, edge_store)[0] >= self.threshold:
                    #     print('pruning error:')
                    #     e.score = scorer.scoring(e, edge_store)[0]
                    #     print(edge)
                    #     print(e)
                    #     raise Exception()
                    halt_flag = False
            # if halt_flag and len(future_edges) > 0:
            #     print("HALT:", str(edge))
            #     pass
        yield None


from random import random


class RandPriorityTraverser(AbstractTraverser):  # todo: make
    def __init__(self, query: Element, target: Element, edge_store: EdgeStore, threshold, *, logger=None):
        self.query = query
        self.target = target
        self.threshold = threshold
        self.edge_queue = hinit(edge_store.iter_edges())
        self.edge_store = edge_store

    def traverse(self) -> Generator:
        logger = get_logger('Alternately.Traverser.traverse')
        while len(self.edge_queue) > 0:
            priority, edge = heapq.heappop(self.edge_queue)
            # if "06/062014/0483" in edge.tnode.code:
            # print("prio:", priority, str(edge))
            if edge.is_leaf_pair():
                parent_qnode = self.rise_query_parent(edge.qnode, stop_before_none=True)
                parent_tnode = self.rise_target_parent(edge.tnode, stop_before_none=True)
                # print('before: ', edge)
                edge = Edge(parent_qnode, parent_tnode)
                if edge.is_leaf_pair():
                    item = yield False  # todo: implement halt(score, edges)
                else:
                    item = yield edge
                    if item is None:
                        continue
                # print('after: ', edge)
            elif edge in self.edge_store or self.edge_store.has_ancestor_pair(edge, include_self=True,
                                                                              filter_mark_tag='delete'):
                # logger.info(str(self.edge_store.find_edge(edge.to_tuple())))
                continue
            # print("prio:", priority, str(edge))
            else:
                # if "06/062014/0483" in edge.tnode.code:
                # print("yield:", priority, str(edge))
                item = yield edge
                if item is None:
                    continue
            score, scorer, edge_store, missing_link = item
            check = False
            if missing_link == 0 and score < self.threshold:
                # check = True
                continue
            parent_qnode = self.get_query_parent(edge.qnode)
            parent_tnode = self.get_target_parent(edge.tnode)
            raised_parent_qnode = self.rise_query_parent(self.get_query_parent(edge.qnode))
            raised_parent_tnode = self.rise_target_parent(self.get_target_parent(edge.tnode))
            if raised_parent_qnode is not None:
                parent_qnode = raised_parent_qnode
            if raised_parent_tnode is not None:
                parent_tnode = raised_parent_tnode
            future_edges = []
            if parent_qnode:
                future_edges.append(Edge(parent_qnode, edge.tnode))
            if parent_tnode:
                future_edges.append(Edge(edge.qnode, parent_tnode))
            if parent_qnode and parent_tnode:
                future_edges.append(Edge(parent_qnode, parent_tnode))
            for e in future_edges:
                upper_limit, lower_limit = scorer.score_range(e, edge_store)
                assert lower_limit <= scorer.scoring(e, edge_store)[0] <= upper_limit, '{0:03}, {1:03}, {2:03}'.format(
                    lower_limit, scorer.scoring(e, edge_store)[0], upper_limit)
                if upper_limit >= self.threshold:
                    # if "06/062014/0483" in e.tnode.code:
                    # print("rise:", e)
                    hpush(self.edge_queue, e, random())
                    # if check and scorer.scoring(e, edge_store)[0] >= self.threshold:
                    #     print('pruning error:')
                    #     e.score = scorer.scoring(e, edge_store)[0]
                    #     print(edge)
                    #     print(e)
                    #     raise Exception()
                    halt_flag = False
            # if halt_flag and len(future_edges) > 0:
            #     print("HALT:", str(edge))
            #     pass
        yield None


class Scorer(object):
    class DistributionObserver(AbstractScorer):
        def __init__(self, scorer):
            super().__init__()
            self.scorer = scorer

        def reset_tmp_cache(self):
            self.scorer.reset_tmp_cache()

        def reset_cache(self):
            self.scorer.reset_cache()

        def scoring(self, edge: Edge, edge_store: EdgeStore, **other_results):
            return self.scorer.scoring(edge, edge_store, **other_results)

        def score_range(self, edge: Edge, edge_store: EdgeStore):
            u, l = self.scorer.score_range(edge, edge_store)
            print(u, l, self.scorer.scoring(edge, edge_store)[0])
            return u, l

    class NoBranchCut(AbstractScorer):
        def __init__(self, scorer):
            super().__init__()
            self.scorer = scorer

        def reset_tmp_cache(self):
            self.scorer.reset_tmp_cache()

        def reset_cache(self):
            self.scorer.reset_cache()

        def scoring(self, edge: Edge, edge_store: EdgeStore, **other_results):
            return self.scorer.scoring(edge, edge_store, **other_results)

        def score_range(self, edge: Edge, edge_store: EdgeStore):
            u, _ = self.scorer.score_range(edge, edge_store)
            return 1.0, 0.0

    class NoPriority(AbstractScorer):
        def __init__(self, scorer):
            super().__init__()
            self.scorer = scorer

        def reset_tmp_cache(self):
            self.scorer.reset_tmp_cache()

        def reset_cache(self):
            self.scorer.reset_cache()

        def scoring(self, edge: Edge, edge_store: EdgeStore, **other_results):
            return self.scorer.scoring(edge, edge_store, **other_results)

        def score_range(self, edge: Edge, edge_store: EdgeStore):
            u, _ = self.scorer.score_range(edge, edge_store)
            return u, 0.0

    class NoPreRead(AbstractScorer):
        def __init__(self, scorer):
            super().__init__()
            self.scorer = scorer

        def reset_tmp_cache(self):
            self.scorer.reset_tmp_cache()

        def reset_cache(self):
            self.scorer.reset_cache()

        def scoring(self, edge: Edge, edge_store: EdgeStore, **other_results):
            return self.scorer.scoring(edge, edge_store, **other_results)

        def score_range(self, edge: Edge, edge_store: EdgeStore):
            _, l = self.scorer.score_range(edge, edge_store)
            return 1.0, l

    class LinearCombination(AbstractScorer):
        def __init__(self, weighted_scorers: list = None):
            super().__init__()
            self.scorers = weighted_scorers or list()

        def __str__(self):
            return " + ".join(["{0}・{1}".format(round(w, 2), s) for s, w in self.scorers])

        def reset_tmp_cache(self):
            for s, _ in self.scorers:
                s.reset_tmp_cache()

        def reset_cache(self):
            # print(self.scorers)
            for s, _ in self.scorers:
                s.reset_cache()

        def add(self, scorer, weight):
            self.scorers.append((scorer, weight))

        def add_batch(self, wscoreres):
            self.scorers.extend(wscoreres)

        def scoring(self, edge: Edge, edge_store: EdgeStore, **other_results):
            score = 0.0
            for scorer, weight in self.scorers:
                single_score, single_other_results = scorer.scoring(edge, edge_store, **other_results)
                score += single_score * weight
                other_results.update(single_other_results)
            return score, other_results

        def score_range(self, edge: Edge, edge_store: EdgeStore):
            uscore = 0.0
            lscore = 0.0
            for scorer, weight in self.scorers:
                u, l = scorer.score_range(edge, edge_store)
                uscore += u * weight
                lscore += l * weight
            # print()
            # print(edge)
            # print("lc:", uscore, lscore)
            return uscore, lscore

    class GeometricMean(AbstractScorer):
        def __init__(self, scorers=None):
            super().__init__()
            self.scorers = scorers or list()

        def __str__(self):
            return "root({})".format("・".join([str(s) for s in self.scorers]))

        def reset_tmp_cache(self):
            for s in self.scorers:
                s.reset_tmp_cache()

        def reset_cache(self):
            for s in self.scorers:
                s.reset_cache()

        def add(self, scorer):
            self.scorers.append(scorer)

        def scoring(self, edge: Edge, edge_store: EdgeStore, **other_results):
            score = 1.0
            for scorer in self.scorers:
                single_score, single_other_results = scorer.scoring(edge, edge_store, **other_results)
                score *= single_score
                other_results.update(single_other_results)
            return math.pow(score, 1.0 / len(self.scorers)), other_results

        def score_range(self, edge: Edge, edge_store: EdgeStore):
            # print(edge)
            uscore = 1.0
            lscore = 1.0
            for scorer in self.scorers:
                u, l = scorer.score_range(edge, edge_store)
                uscore *= u
                lscore *= l
            # print('geo:', math.pow(uscore, 1.0 / len(self.scorers)),uscore,math.pow(lscore, 1.0 / len(self.scorers)), lscore)
            return math.pow(uscore, 1.0 / len(self.scorers)), math.pow(lscore, 1.0 / len(self.scorers))

    # todo: create query weight scorer
    @staticmethod
    def leaf_simple_match(edge, edge_store, other_results):
        matching = list(
            e for e in edge_store.iter_edges(qkey=edge.qnode.code, tkey=edge.tnode.code) if e.is_leaf_pair()
        )
        other_results['leaf_simple_match'] = matching
        return matching

    class QueryLeafCoverage(AbstractScorer):
        def __init__(self, maxmatch: bool = False):
            super().__init__()
            self.maxmatch = maxmatch

        def __str__(self):
            return "QLC"

        def reset_tmp_cache(self):
            super().reset_tmp_cache()
            self._tmp_cache.update({'nume': {}})

        def reset_cache(self):
            super().reset_cache()

        def scoring(self, edge: Edge, edge_store: EdgeStore, **_calc_cache):
            logger = get_logger('Alternately.Scorer.QueryLeafCoverage.scoring')
            if self.maxmatch:
                matching = _calc_cache.get('leaf_max_match', None) or edge_store.leaf_max_match(edge)
            else:
                matching = _calc_cache.get('leaf_simple_match', None) or Scorer.leaf_simple_match(edge, edge_store,
                                                                                                  _calc_cache)
                matching = set(e.qnode.code for e in matching)
            match_num = sum(1 for e in matching)
            qleaf_num = self.calc_qleaf_count(edge)
            score = match_num / qleaf_num
            logger.debug('%f/%f = %f', match_num, qleaf_num, score)
            self.caching('nume', edge, len(matching))
            return score, _calc_cache

        def score_range(self, edge: Edge, edge_store: EdgeStore):
            leaf_num = self.calc_qleaf_count(edge)
            if self.maxmatch:
                used_leaf_num = len(set(
                    [e.qnode.code for e in edge_store.iter_edges(qkey=edge.qnode, tkey=edge.tnode) if e.is_leaf_pair()]
                ))
                return used_leaf_num / leaf_num, max([self._cache['nume'].get(c, 1) for c in edge.tnode]) / leaf_num
            else:
                match_leaf_num = self.get_match_qleaf_num(edge, edge_store)
                upper_limit = min(match_leaf_num / leaf_num, 1)
                lower_limit = max([self.get_from_cache('nume', c, 1) for c in edge.qnode]) / leaf_num
                return upper_limit, lower_limit

    class TargetLeafCoverage(AbstractScorer):
        def __init__(self, maxmatch: bool = False):
            super().__init__()
            self.maxmatch = maxmatch

        def __str__(self):
            return "TLC"

        def reset_tmp_cache(self):
            super().reset_tmp_cache()
            self._tmp_cache.update({'nume': {}})

        def reset_cache(self):
            super().reset_cache()

        def scoring(self, edge: Edge, edge_store: EdgeStore, **_calc_cache):
            logger = get_logger('Alternately.Scorer.QueryLeafCoverage.scoring')
            if self.maxmatch:
                matching = _calc_cache.get('leaf_max_match', None) or edge_store.leaf_max_match(edge)
            else:
                matching = _calc_cache.get('leaf_simple_match', None) or Scorer.leaf_simple_match(edge, edge_store,
                                                                                                  _calc_cache)
                matching = set(e.tnode.code for e in matching)
            match_num = sum(1 for e in matching)
            tleaf_num = self.calc_tleaf_count(edge)
            score = match_num / tleaf_num
            logger.debug('%f/%f = %f', match_num, tleaf_num, score)
            self.caching('nume', edge, len(matching))
            return score, _calc_cache

        def score_range(self, edge: Edge, edge_store: EdgeStore):
            leaf_num = self.calc_tleaf_count(edge)
            if self.maxmatch:
                used_leaf_num = len(set(
                    [e.tnode.code for e in edge_store.iter_edges(qkey=edge.qnode, tkey=edge.tnode) if e.is_leaf_pair()]
                ))
                return used_leaf_num / leaf_num, max([self._cache['nume'].get(c, 1) for c in edge.tnode]) / leaf_num
            else:
                match_leaf_num = self.get_match_tleaf_num(edge, edge_store)
                upper_limit = min(match_leaf_num / leaf_num, 1)
                lower_limit = max([self.get_from_cache('nume', c, 1) for c in edge.tnode]) / leaf_num
                return upper_limit, lower_limit

    class QueryLeafProximity(AbstractScorer):
        def __init__(self, maxmatch: bool = False):
            super().__init__()
            self.maxmatch = maxmatch

        def __str__(self):
            return "QLP{}".format("_max" if self.maxmatch else "")

        def reset_tmp_cache(self):
            super().reset_tmp_cache()
            self._tmp_cache.update({'nume': {}})

        def reset_cache(self):
            super().reset_cache()

        def scoring(self, edge: Edge, edge_store: EdgeStore, **other_results):
            logger = get_logger('Alternately.Scorer.QueryLeafCoverage.scoring')
            if self.maxmatch:
                matching = other_results.get('leaf_max_match', None) or edge_store.leaf_max_match(edge)
                other_results['leaf_max_match'] = matching
            else:
                matching = other_results.get('leaf_simple_match', None) or Scorer.leaf_simple_match(edge, edge_store,
                                                                                                    other_results)
            proximity_list = list()
            current_proximity = 0.0
            for e in edge.qnode.iterXsentence_elem():
                for edge in matching:
                    if e.code == edge.qnode.code:
                        proximity_list.append(current_proximity)
                        break
                else:
                    current_proximity += 1  # / (1 + e.LEVEL - etypes.ParagraphSentence.LEVEL)
            proximity_list.append(current_proximity)
            score = 1 / (1 + max(proximity_list))
            logger.debug('1/(1+%f) = %f', max(proximity_list), score)
            self.caching('nume', edge, len(matching))
            return score, other_results

        def score_range(self, edge: Edge, edge_store: EdgeStore):
            if edge.qnode.CATEGORY == etypes.CATEGORY_TEXT:
                return 1, 1
            leaf_num = self.calc_qleaf_count(edge)
            match_leaf_num = self.get_match_qleaf_num(edge, edge_store)
            max_prox = leaf_num - max([self.get_from_cache('nume', c, 1) for c in edge.qnode])
            min_prox = (leaf_num - match_leaf_num) // (match_leaf_num + 1) + 1 if leaf_num > match_leaf_num else 0
            # print('Q:', max_prox, min_prox)
            return 1 / (1 + min_prox), 1 / (1 + max_prox)

    class TargetLeafProximity(AbstractScorer):
        def __init__(self, maxmatch: bool = False):
            super().__init__()
            self.maxmatch = maxmatch

        def __str__(self):
            return "TLP{}".format("_max" if self.maxmatch else "")

        def reset_tmp_cache(self):
            super().reset_tmp_cache()
            self._tmp_cache.update({'nume': {}})

        def reset_cache(self):
            super().reset_cache()

        def scoring(self, edge: Edge, edge_store: EdgeStore, **other_results):
            logger = get_logger('Alternately.Scorer.QueryLeafCoverage.scoring')
            if self.maxmatch:
                matching = other_results.get('leaf_max_match', None) or edge_store.leaf_max_match(edge)
                other_results['leaf_max_match'] = matching
            else:
                matching = other_results.get('leaf_simple_match', None) or Scorer.leaf_simple_match(edge, edge_store,
                                                                                                    other_results)
            proximity_list = list()
            current_proximity = 0.0
            for e in edge.tnode.iterXsentence_elem():
                for edge in matching:
                    if e.code == edge.tnode.code:
                        proximity_list.append(current_proximity)
                        break
                else:
                    current_proximity += 1  # / (1 + e.LEVEL - etypes.ParagraphSentence.LEVEL)
            proximity_list.append(current_proximity)
            score = 1 / (1 + max(proximity_list))
            logger.debug('1/(1+%f) = %f', max(proximity_list), score)
            self.caching('nume', edge, len(matching))
            return score, other_results

        def score_range(self, edge: Edge, edge_store: EdgeStore):
            if edge.tnode.CATEGORY == etypes.CATEGORY_TEXT:
                return 1, 1
            # used_leaf_num = len(set(
            #    [e.tnode.code for e in edge_store.iter_edges(qkey=edge.qnode, tkey=edge.tnode) if e.is_leaf_pair()]
            # ))
            leaf_num = self.calc_tleaf_count(edge)
            match_leaf_num = self.get_match_tleaf_num(edge, edge_store)
            max_prox = leaf_num - max([self.get_from_cache('nume', c, 1) for c in edge.tnode])
            min_prox = (leaf_num - match_leaf_num) // (match_leaf_num + 1) + 1 if leaf_num > match_leaf_num else 0
            # print('T:', max_prox, min_prox)
            return 1 / (1 + min_prox), 1 / (1 + max_prox)

    class QueryLeafMoment(AbstractScorer):
        def __init__(self, moment_n, use_logn=True, maxmatch: bool = False):
            super().__init__()
            self.moment_n = moment_n
            self.maxmatch = maxmatch
            self.use_logn = use_logn

        def __str__(self):
            return "QL{}M".format(self.moment_n)

        def reset_tmp_cache(self):
            super().reset_tmp_cache()
            self._tmp_cache.update({'nume': {}})

        def reset_cache(self):
            super().reset_cache()

        def scoring(self, edge: Edge, edge_store: EdgeStore, **other_results):
            logger = get_logger('Alternately.Scorer.QueryLeafCoverage.scoring')
            if self.maxmatch:
                matching = other_results.get('leaf_max_match', None) or edge_store.leaf_max_match(edge)
                other_results['leaf_max_match'] = matching
            else:
                matching = other_results.get('leaf_simple_match', None) or Scorer.leaf_simple_match(edge, edge_store,
                                                                                                    other_results)
            coordinates = list()
            match_codes = set([e.qnode.code for e in matching])
            for x, e in enumerate(edge.qnode.iterXsentence_elem()):
                if e.code not in match_codes:
                    coordinates.append(x)
            m = moment(np.array(coordinates), moment=self.moment_n)
            if self.use_logn:
                return 1 / (1 + math.log(m + 1)), other_results
            else:
                return 1 / (1 + m), other_results

    class TargetLeafMoment(AbstractScorer):
        def __init__(self, moment_n, use_logn=True, maxmatch: bool = False):
            super().__init__()
            self.moment_n = moment_n
            self.maxmatch = maxmatch
            self.use_logn = use_logn

        def __str__(self):
            return "TL{}M".format(self.moment_n)

        def reset_tmp_cache(self):
            super().reset_tmp_cache()
            self._tmp_cache.update({'nume': {}})

        def reset_cache(self):
            super().reset_cache()

        def scoring(self, edge: Edge, edge_store: EdgeStore, **other_results):
            logger = get_logger('Alternately.Scorer.QueryLeafCoverage.scoring')
            if self.maxmatch:
                matching = other_results.get('leaf_max_match', None) or edge_store.leaf_max_match(edge)
                other_results['leaf_max_match'] = matching
            else:
                matching = other_results.get('leaf_simple_match', None) or Scorer.leaf_simple_match(edge, edge_store,
                                                                                                    other_results)
            coordinates = list()
            match_codes = set([e.tnode.code for e in matching])
            for x, e in enumerate(edge.tnode.iterXsentence_elem()):
                if e.code not in match_codes:
                    coordinates.append(x)
            if len(coordinates) == 0:
                return 0, other_results
            m = moment(np.array(coordinates), moment=self.moment_n)
            if self.use_logn:
                return 1 / (1 + math.log(m + 1)), other_results
            else:
                return 1 / (1 + m), other_results

    class Balancer(AbstractScorer):
        def __init__(self):
            super().__init__()

        def scoring(self, edge: Edge, edge_store: EdgeStore, **other_results):
            tleaf_count = self.calc_tleaf_count(edge)
            qleaf_count = self.calc_qleaf_count(edge)
            if tleaf_count < qleaf_count:
                return tleaf_count / qleaf_count, other_results
            else:
                return qleaf_count / tleaf_count, other_results

        def score_range(self, edge: Edge, edge_store: EdgeStore):
            s = self.scoring(edge, edge_store)[0]
            return s, s

    class LCA(AbstractScorer):
        def scoring(self, edge: Edge, edge_store: EdgeStore, **other_results):
            logger = get_logger('Alternately.Scorer.LCA.scoring')
            for e in edge_store.iter_edges(qkey=edge.qnode.code, tkey=edge.tnode.code):
                if e.is_leaf_pair() or e.score > 1.5 or e == edge:
                    continue
                # print('%s is not an LCA pair (too big)', str(edge))
                return 0.0, other_results
            if not edge_store.is_lca_pair_edge(edge):
                # print('%s is not an LCA pair (too small)', str(edge))
                return 2.0, other_results
            print('%s is an LCA pair.', str(edge))
            return 1.0, other_results

        def score_range(self, edge: Edge, edge_store: EdgeStore):
            return 1.0, 1.0

    class LimitLayer(AbstractScorer):
        def __init__(self, scorer=None, layer="Article"):
            super().__init__()
            self.scorer = scorer
            self.layer = layer

        def add(self, scorer):
            self.scorer = scorer

        def reset_tmp_cache(self):
            self.scorer.reset_tmp_cache()

        def reset_cache(self):
            self.scorer.reset_cache()

        def scoring(self, edge: Edge, edge_store: EdgeStore, **other_results):
            if self.layer not in edge.qnode.code or self.layer not in edge.tnode.code:
                return -1.0, other_results
            return self.scorer.scoring(edge, edge_store, **other_results)

        def score_range(self, edge: Edge, edge_store: EdgeStore):
            if self.layer not in edge.qnode.code or self.layer not in edge.tnode.code:
                return -1.0, -1.0
            return self.scorer.score_range(edge, edge_store)
