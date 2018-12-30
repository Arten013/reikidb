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


class PriorityTraverser(AbstractTraverser):  # todo: make
    def __init__(self, query: Element, target: Element, edge_store: EdgeStore, *, logger=None):
        self.query = query
        self.target = target
        self.edge_queue = PriorityQueue()
        self.edge_store = edge_store
        for e in edge_store.iter_edges():
            self.edge_queue.put(e)

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


class Scorer(object):
    class LinearCombination(AbstractScorer):
        def __init__(self, weighted_scorers: list):
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

    class GeometricMean(AbstractScorer):
        def __init__(self, scorers):
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

    # todo: create query weight scorer
    @staticmethod
    def leaf_simple_match(edge, edge_store, other_results):
        matching = list(
            e for e in edge_store.iter_edges(qkey=edge.qnode.code, tkey=edge.tnode.code) if e.is_leaf_pair()
        )
        other_results['leaf_simple_match'] = matching
        return matching

    class QueryLeafCoverage(AbstractScorer):
        def __init__(self, using_qweight: bool = False, maxmatch: bool = False):
            self.maxmatch = maxmatch
            self.using_qweight = using_qweight

        def scoring(self, edge: Edge, edge_store: EdgeStore, **_calc_cache):
            logger = get_logger('Alternately.Scorer.QueryLeafCoverage.scoring')
            if self.maxmatch:
                matching = _calc_cache.get('leaf_max_match', None) or edge_store.leaf_max_match(edge)
            else:
                matching = _calc_cache.get('leaf_simple_match', None) or Scorer.leaf_simple_match(edge, edge_store,
                                                                                                  _calc_cache)
                matching = set(e.qnode.code for e in matching)
            if self.using_qweight:
                match_num = sum(e.qnode.attrib.get('weight', 1.0) for e in matching)
                qleaf_num = sum(edge.qnode.find_by_code(c).attrib.get('weight', 1.0)
                                for c in edge.qnode.iterXsentence_code())
            else:
                match_num = len(matching)
                qleaf_num = sum(1 for _ in edge.qnode.iterXsentence(include_code=False, include_value=False))
            score = match_num / qleaf_num
            logger.debug('%f/%f = %f', match_num, qleaf_num, score)
            return score, _calc_cache

    class TargetLeafCoverage(AbstractScorer):
        def __init__(self, maxmatch: bool = False):
            self.maxmatch = maxmatch

        def scoring(self, edge: Edge, edge_store: EdgeStore, **other_results):
            logger = get_logger('Alternately.Scorer.TargetLeafCoverage.scoring')
            tleaf_num = sum(1 for _ in edge.tnode.iterXsentence(include_code=False, include_value=False))
            if self.maxmatch:
                matching = other_results.get('leaf_max_match', None) or edge_store.leaf_max_match(edge)
                other_results['leaf_max_match'] = matching
            else:
                matching = other_results.get('leaf_simple_match', None) or Scorer.leaf_simple_match(edge, edge_store,
                                                                                                    other_results)
                matching = set(e.tnode.code for e in matching)
            match_num = len(matching)
            score = match_num / tleaf_num
            logger.debug('%f/%f = %f', match_num, tleaf_num, score)
            return score, other_results

    class QueryLeafProximity(AbstractScorer):
        def __init__(self, using_qweight=False, maxmatch: bool = False):
            self.using_qweight = using_qweight
            self.maxmatch = maxmatch

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
                    if self.using_qweight:
                        current_proximity += e.attrib.get('weight', 1.0) * 1 / (
                                    1 + e.LEVEL - etypes.ParagraphSentence.LEVEL)
                    else:
                        current_proximity += 1 / (1 + e.LEVEL - etypes.ParagraphSentence.LEVEL)
            proximity_list.append(current_proximity)
            score = 1 / (1 + max(proximity_list))
            logger.debug('1/(1+%f) = %f', max(proximity_list), score)
            return score, other_results

    class TargetLeafProximity(AbstractScorer):
        def __init__(self, maxmatch: bool = False):
            self.maxmatch = maxmatch

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
                    current_proximity += 1 / (1 + e.LEVEL - etypes.ParagraphSentence.LEVEL)
            proximity_list.append(current_proximity)
            score = 1 / (1 + max(proximity_list))
            logger.debug('1/(1+%f) = %f', max(proximity_list), score)
            return score, other_results

    class LCA(AbstractScorer):
        def scoring(self, edge: Edge, edge_store: EdgeStore, **other_results):
            logger = get_logger('Alternately.Scorer.LCA.scoring')
            for e in edge_store.iter_edges(qkey=edge.qnode.code, tkey=edge.tnode.code):
                if e.is_leaf_pair() or e.score == 0:
                    continue
                print('%s is not an LCA pair (too big)', str(edge))
                return 0.0, other_results
            if not edge_store.is_lca_pair_edge(edge):
                print('%s is not an LCA pair (too small)', str(edge))
                return 0.0, other_results
            print('%s is an LCA pair.', str(edge))
            return 1.0, other_results
