from jstatutree.element import Element
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
            #if "06/062014/0483" in edge.tnode.code:
                #print("prio:", priority, str(edge))
            if edge.is_leaf_pair():
                parent_qnode = self.rise_query_parent(edge.qnode, stop_before_none=True)
                parent_tnode = self.rise_target_parent(edge.tnode, stop_before_none=True)
                #print('before: ', edge)
                edge = Edge(parent_qnode, parent_tnode)
                if edge.is_leaf_pair():
                    item = yield False  # todo: implement halt(score, edges)
                else:
                    item = yield edge
                    if item is None:
                        continue
                #print('after: ', edge)
            elif self.edge_store.has_ancestor_pair(edge, include_self=True):
                # logger.info(str(self.edge_store.find_edge(edge.to_tuple())))
                continue
            #print("prio:", priority, str(edge))
            else:
                #if "06/062014/0483" in edge.tnode.code:
                    #print("yield:", priority, str(edge))
                item = yield edge
                if item is None:
                    continue
            parent_qnode = self.get_query_parent(edge.qnode)
            parent_tnode = self.get_target_parent(edge.tnode)
            raised_parent_qnode = self.rise_query_parent(self.get_query_parent(edge.qnode))
            raised_parent_tnode = self.rise_target_parent(self.get_target_parent(edge.tnode))
            if raised_parent_qnode is not None:
                parent_qnode = raised_parent_qnode
            if raised_parent_tnode is not None:
                parent_tnode = raised_parent_tnode
            scorer, edge_store = item
            future_edges = []
            if parent_qnode:
                future_edges.append(Edge(parent_qnode, edge.tnode))
            if parent_tnode:
                future_edges.append(Edge(edge.qnode, parent_tnode))
            if parent_qnode and parent_tnode:
                future_edges.append(Edge(parent_qnode, parent_tnode))
            halt_flag = True
            for e in future_edges:
                upper_limit, lower_limit = scorer.score_range(e, edge_store)
                if upper_limit >= self.threshold:
                    #if "06/062014/0483" in e.tnode.code:
                        #print("rise:", e)
                    hpush(self.edge_queue, e, -lower_limit)
                    halt_flag = False
            if halt_flag and len(future_edges) > 0:
                #print("HALT:", str(edge))
                pass
        yield None


class Scorer(object):
    class LinearCombination(AbstractScorer):
        def __init__(self, weighted_scorers: list):
            super().__init__()
            self.scorers = weighted_scorers

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
            #print()
            #print(edge)
            #print("lc:", uscore, lscore)
            return uscore, lscore

    class GeometricMean(AbstractScorer):
        def __init__(self, scorers):
            super().__init__()
            self.scorers = scorers

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
            uscore = 1.0
            lscore = 1.0
            for scorer in self.scorers:
                u, l = scorer.score_range(edge, edge_store)
                uscore *= u
                lscore *= l
            #print('geo:', math.pow(uscore, 1.0 / len(self.scorers)),uscore,math.pow(lscore, 1.0 / len(self.scorers)), lscore)
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
            self._cache.update({'score': {}, 'nume': {}})

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
            self._cache['nume'][edge] = match_num
            return score, _calc_cache

        def score_range(self, edge: Edge, edge_store: EdgeStore):
            qcodes = []
            for e in edge_store.iter_edges(qkey=edge.qnode, tkey=edge.tnode):
                if not e.is_leaf_pair():
                    continue
                qcodes.append(e.qnode.code)
            num = len(set(qcodes))
            deno = self.calc_qleaf_count(edge)
            return num/deno, max([self._cache['nume'].get(c, 1) for c in edge.qnode])/deno

    class TargetLeafCoverage(AbstractScorer):
        def __init__(self, maxmatch: bool = False):
            super().__init__()
            self.maxmatch = maxmatch
            self._cache.update({'score': {}, 'nume': {}})

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
            self._cache['nume'][edge] = match_num
            return score, _calc_cache

        def score_range(self, edge: Edge, edge_store: EdgeStore):
            tcodes = []
            for e in edge_store.iter_edges(qkey=edge.qnode, tkey=edge.tnode):
                if not e.is_leaf_pair():
                    continue
                tcodes.append(e.tnode.code)
            num = len(set(tcodes))
            deno = self.calc_tleaf_count(edge)
            return num/deno, max([self._cache['nume'].get(c, 1) for c in edge.tnode])/deno


    class QueryLeafProximity(AbstractScorer):
        def __init__(self, maxmatch: bool = False):
            super().__init__()
            self.maxmatch = maxmatch
            self._cache.update({'score': {}, 'nume': {}})

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
                    current_proximity += 1 #/ (1 + e.LEVEL - etypes.ParagraphSentence.LEVEL)
            proximity_list.append(current_proximity)
            score = 1 / (1 + max(proximity_list))
            logger.debug('1/(1+%f) = %f', max(proximity_list), score)
            self._cache['nume'][edge] = len(matching)
            return score, other_results

        def score_range(self, edge: Edge, edge_store: EdgeStore):
            qcodes = []
            for e in edge_store.iter_edges(qkey=edge.qnode, tkey=edge.tnode):
                if not e.is_leaf_pair():
                    continue
                qcodes.append(e.qnode.code)
            num = len(set(qcodes))
            deno = self.calc_qleaf_count(edge)
            max_prox = deno - max([self._cache['nume'].get(c, 1) for c in edge.qnode])
            min_prox = (deno - num)//(num+1)+1 if deno > num else 0.0
            return 1/(1+min_prox), 1/(1+max_prox)

    class TargetLeafProximity(AbstractScorer):
        def __init__(self, maxmatch: bool = False):
            super().__init__()
            self.maxmatch = maxmatch
            self._cache.update({'score': {}, 'nume': {}})

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
                    current_proximity += 1 #/ (1 + e.LEVEL - etypes.ParagraphSentence.LEVEL)
            proximity_list.append(current_proximity)
            score = 1 / (1 + max(proximity_list))
            logger.debug('1/(1+%f) = %f', max(proximity_list), score)
            self._cache['nume'][edge] = len(matching)
            return score, other_results

        def score_range(self, edge: Edge, edge_store: EdgeStore):
            tcodes = []
            for e in edge_store.iter_edges(qkey=edge.qnode, tkey=edge.tnode):
                if not e.is_leaf_pair():
                    continue
                tcodes.append(e.tnode.code)
            num = len(set(tcodes))
            deno = self.calc_tleaf_count(edge)
            max_prox = deno - max([self._cache['nume'].get(c, 1) for c in edge.tnode])
            min_prox = (deno - num)//(num+1)+1 if deno > num else 0.0
            return 1/(1+min_prox), 1/(1+max_prox)

    class LCA(AbstractScorer):
        def scoring(self, edge: Edge, edge_store: EdgeStore, **other_results):
            logger = get_logger('Alternately.Scorer.LCA.scoring')
            for e in edge_store.iter_edges(qkey=edge.qnode.code, tkey=edge.tnode.code):
                if e.is_leaf_pair() or e.score > 1.5 or e == edge:
                    continue
                #print('%s is not an LCA pair (too big)', str(edge))
                return 0.0, other_results
            if not edge_store.is_lca_pair_edge(edge):
                #print('%s is not an LCA pair (too small)', str(edge))
                return 2.0, other_results
            print('%s is an LCA pair.', str(edge))
            return 1.0, other_results

        def score_range(self, edge: Edge, edge_store: EdgeStore):
            return 1.0, 1.0