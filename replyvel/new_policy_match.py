import jstatutree
from jstatutree.element import Element
from jstatutree import Jstatutree
from pathlib import Path
from typing import Generator, Mapping, Union, Sequence
import math
import numpy as np
from collections import OrderedDict
from .utils.npm_abstracts import AbstractScorer, AbstractActivator, AbstractTraverser
from .utils.edge import Edge, EdgeStore
from .utils.logger import get_logger
from graphviz import Digraph


class PolicyMatchObject(object):
    def __init__(self):
        self.units = {}

    def add_unit(self, query_tree: Jstatutree, match_tree: Jstatutree, edge_store: EdgeStore):
        self.units[match_tree.lawdata.code] = PolicyMatchUnit(query_tree, match_tree, edge_store)


class PolicyMatchUnit(object):
    def __init__(self, query_tree: Jstatutree, match_tree: Jstatutree, edge_store: EdgeStore):
        self.query_tree = query_tree
        self.match_tree = match_tree
        self.edge_store = edge_store

    def _tree_name_factory(self, elem: Element) -> str:
        if str(self.query_tree.lawdata.code) in str(elem.code):
            return self.query_tree.lawdata.name
        return self.match_tree.lawdata.name

    def find_matching_edge(self, threshold: float) -> list:
        candidates = self.edge_store.filtering(key=lambda x: x.score >= threshold)
        results = []
        for e in sorted(candidates, key=lambda e: str(e.qnode.code).count('/')**2+str(e.tnode.code).count('/')**2):
            for r in results:
                if r.qnode.code in e.qnode.code:
                    break
            else:
                results.append(e)
        return results

    def comp_table(self, threshold: float, **kwargs) -> Digraph:
        code_pairs = [e.to_tuple(node_as_code=True) for e in self.find_matching_edge(threshold)]
        print(code_pairs[0])
        sub_code_pairs = [
            (sub_edge.qnode.code, sub_edge.tnode.code, sub_edge.score)
            for qcode, tcode, _ in code_pairs
            for sub_edge in self.edge_store.leaf_max_match((qcode, tcode)) if sub_edge.is_leaf_pair()
        ]
        # qelem = self.bind_by_lca(*qcodes) if len(qcodes) else self.bind_by_lca(*self.queries.keys())
        return jstatutree.graph.cptable_graph(self.query_tree.getroot(), [self.match_tree.getroot()], code_pairs,
                                              sub_code_pairs,
                                              tree_name_factory=self._tree_name_factory, **kwargs)


class SimilarityActivator:
    class Void(AbstractActivator):
        def activate_core(self, val: float) -> float:
            return val

    class Sigmoid(AbstractActivator):
        def __init__(self, gain: int, center: float):
            self.gain = gain
            self.center = center
            super().__init__()

        def activate_core(self, val: float) -> float:
            return 1 / (1 + math.exp((self.center - val) * self.gain))


class PolicyMatchFactory(object):
    def __init__(self, query_tree: Jstatutree, tree_factory):
        self.query_tree = query_tree
        self.self = EdgeStore()
        self.tree_store = {}
        self.tree_factory = tree_factory
        self.leaf_edge_store = EdgeStore()

    def matching(self, scorer, traverser_cls, activator, * , least_matching_node=0):
        logger = get_logger('PolicyMatchFactory.matching')
        logger.info('matching() called')
        obj = PolicyMatchObject()
        for target_tree, matching_edge_store in self.scoring(scorer, traverser_cls, activator, least_matching_node=least_matching_node):
            obj.add_unit(self.query_tree, target_tree, matching_edge_store)
        logger.info('matching() finished')
        return obj

    def add_leaf(self, qcode: str, tcode: str, similarity: float):
        logger = get_logger('PolicyMatchFactory.add_leaf')
        qnode = self.query_tree.find_by_code(qcode)
        rcode = tcode[:14]
        if rcode not in self.tree_store:
            self.tree_store[rcode] = self.tree_factory(rcode)
        tnode = self.tree_store[rcode].find_by_code(tcode)
        logger.debug("add edge %s %s %f", str(qnode.code), str(tnode.code), similarity)
        self.leaf_edge_store.add_edge(qnode, tnode, similarity)

    def scoring(self, traverser_cls, scorer: AbstractScorer, activator: AbstractActivator, *, least_matching_node=0):
        logger = get_logger('PolicyMatchFactory.scoring')
        for target_code, target_tree in self.tree_store.items():
            if least_matching_node > 0 and \
                    len(list(self.leaf_edge_store.iter_edges(tkey=target_code))) < least_matching_node:
                logger.info('skip(single node matching): %s', str(target_code))
                continue
            logger.info('scoring %s', target_tree.lawdata.code)
            target = target_tree.getroot()
            edges = list(self.leaf_edge_store.iter_edges(tkey=target.code))
            reformatted_edges = scorer.reformatting_leaf_edges(edges)
            matching_edge_store = activator.initial_edges(*reformatted_edges)
            logger.debug('initial edge store size: %d', len(matching_edge_store))
            traverser = traverser_cls(self.query_tree.getroot(), target, matching_edge_store).traverse()
            edge = next(traverser)
            while True:
                if edge is None:
                    break
                edge.score = scorer.scoring(edge, matching_edge_store)[0]
                matching_edge_store.add_edge(*edge.to_tuple())
                logger.info('add edge: %s', str(edge))
                edge = traverser.send(edge.score)
            logger.info('finished scoring %s', target_tree.lawdata.code)
            yield target_tree, matching_edge_store
        logger.debug('quit')


class Alternately:
    class Traverser(AbstractTraverser):
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

        def traverse(self) -> Generator:
            logger = get_logger('Alternately.Traverser.traverse')
            logger.debug('Initial queue size: %d', self.edge_queue.qsize())
            while not self.edge_queue.empty():
                edge = self.edge_queue.get()
                if not edge.is_leaf_pair():
                    if edge in self.edge_store:
                        logger.debug('skip: already scored')
                        #logger.info(str(self.edge_store.find_edge(edge.to_tuple())))
                        continue
                    logger.debug('yield edge %s', str(edge))
                    score = yield edge # todo: implement halt(score, edges)
                parent_qnode = self.get_query_parent(edge.qnode)
                parent_tnode = self.get_target_parent(edge.tnode)
                if parent_qnode:
                    self.edge_queue.put(Edge(parent_qnode, edge.tnode, 0))
                if parent_tnode:
                    self.edge_queue.put(Edge(edge.qnode, parent_tnode, 0))
                if parent_qnode and parent_tnode:
                    # self.edge_queue.put(Edge(parent_qnode, parent_tnode, 0))
                    pass
            yield None


    class Scorer(object):
        class ParallelScorer(AbstractScorer):
            def __init__(self):
                self.scorers = OrderedDict()

            @property
            def size(self):
                return len(self.scorers)

            @property
            def is_parallel(self):
                return True

            def reformatting_leaf_edges(self, *edges: Sequence[Edge]) -> Sequence[Edge]:
                for e in edges:
                    if not isinstance(e, np.ndarray):
                        e.score = np.array([e.score]*self.size)
                return edges

            def add(self, tag: str, scorer):
                self.scorers[tag] = scorer

            def add_by_keyvalue(self, keyvalue: Mapping):
                self.scorers.update(keyvalue)

            def scoring(self, edge: Edge, edge_store: EdgeStore, **other_results):
                score_list = []
                for scorer in self.scorers:
                    single_score, single_other_results = scorer.scoring(edge, edge_store, **other_results)
                    score_list += [single_score]
                    other_results.update(single_other_results)
                return np.array(score_list), other_results

        class LinearCombination(AbstractScorer):
            def __init__(self):
                self.scorers = []

            def add(self, scorer, weight):
                self.scorers.append((scorer, weight))

            def scoring(self, edge:Edge, edge_store: EdgeStore, **other_results):
                score = 0.0
                for scorer, weight in self.scorers:
                    single_score, single_other_results = scorer.scoring(edge, edge_store, **other_results)
                    score += single_score * weight
                    other_results.update(single_other_results)
                return score, other_results

        class GeometricMean(AbstractScorer):
            def __init__(self):
                self.scorers = []

            def add(self, scorer):
                self.scorers.append(scorer)

            def scoring(self, edge:Edge, edge_store: EdgeStore, **other_results):
                score = 1.0
                for scorer in self.scorers:
                    single_score, single_other_results = scorer.scoring(edge, edge_store, **other_results)
                    score *= single_score
                    other_results.update(single_other_results)
                return math.pow(score, 1.0/len(self.scorers)), other_results

        class QueryLeafCoverage(AbstractScorer):
            def __init__(self, using_qweight=False):
                self.using_qweight = using_qweight

            def scoring(self, edge:Edge, edge_store: EdgeStore, **other_results):
                logger = get_logger('Alternately.Scorer.QueryLeafCoverage.scoring')
                if self.using_qweight:
                    qleaf_num = sum(edge.qnode.find_by_code(c).attrib.get('weight', 1.0)
                                    for c in edge.qnode.iterXsentence_code())
                else:
                    qleaf_num = sum(1 for _ in edge.qnode.iterXsentence(include_code=False, include_value=False))
                matching = other_results.get('leaf_max_match', None) or edge_store.leaf_max_match(edge)
                other_results['leaf_max_match'] = matching
                match_num = len(matching)
                score = match_num / qleaf_num
                logger.debug('%d/%d = %f',match_num, qleaf_num, score)
                return score, other_results

        class TargetLeafCoverage(AbstractScorer):
            def scoring(self, edge:Edge, edge_store: EdgeStore, **other_results):
                logger = get_logger('Alternately.Scorer.TargetLeafCoverage.scoring')
                tleaf_num = sum(1 for _ in edge.tnode.iterXsentence(include_code=False, include_value=False))
                matching = other_results.get('leaf_max_match', None) or edge_store.leaf_max_match(edge)
                other_results['leaf_max_match'] = matching
                match_num = len(matching)
                score = match_num/tleaf_num
                logger.debug('%d/%d = %f',match_num, tleaf_num, score)
                return score, other_results

        class QueryLeafProximity(AbstractScorer):
            def __init__(self, using_qweight=False):
                self.using_qweight = using_qweight

            def scoring(self, edge:Edge, edge_store: EdgeStore, **other_results):
                logger = get_logger('Alternately.Scorer.QueryLeafCoverage.scoring')
                matching = other_results.get('leaf_max_match', None) or edge_store.leaf_max_match(edge)
                other_results['leaf_max_match'] = matching
                proximity_list = list()
                current_proximity = 0.0
                for e in edge.qnode.iterXsentence_elem():
                    for edge in matching:
                        if e.code == edge.qnode.code:
                            proximity_list.append(current_proximity)
                            break
                    else:
                        if self.using_qweight:
                            current_proximity += e.attrib.get('weight', 1.0)
                        else:
                            current_proximity += 1
                proximity_list.append(current_proximity)
                score = 1/(1+max(proximity_list))
                logger.debug('1/(1+%d) = %f',max(proximity_list), score)
                return score, other_results

        class TargetLeafProximity(AbstractScorer):
            def scoring(self, edge:Edge, edge_store: EdgeStore, **other_results):
                logger = get_logger('Alternately.Scorer.QueryLeafCoverage.scoring')
                matching = other_results.get('leaf_max_match', None) or edge_store.leaf_max_match(edge)
                other_results['leaf_max_match'] = matching
                proximity_list = list()
                current_proximity = 0.0
                for e in edge.tnode.iterXsentence_elem():
                    for edge in matching:
                        if e.code == edge.tnode.code:
                            proximity_list.append(current_proximity)
                            break
                    else:
                        current_proximity += 1
                proximity_list.append(current_proximity)
                score = 1/(1+max(proximity_list))
                logger.debug('1/(1+%d) = %f',max(proximity_list), score)
                return score, other_results


