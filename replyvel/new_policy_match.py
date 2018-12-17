import abc
import pygtrie as trie
from collections import Counter
import jstatutree
from jstatutree.element import Element
from jstatutree import Jstatutree
from pathlib import Path
from typing import Generator
import math
from queue import Queue
import networkx as nx
from networkx.algorithms import bipartite
import logging
import time



_default_handler = logging.StreamHandler()
_default_handler.setLevel(logging.DEBUG)
_default_formatter = logging.Formatter('{name} {levelname:8s} {message}',
                        style='{')
_default_handler.setFormatter(_default_formatter)
def _get_logger(module_name):
    logger = logging.getLogger(module_name)
    logger.setLevel(logging.INFO)
    logger.addHandler(_default_handler)
    logger.propagate = True
    return logger

class Edge(object):
    def __init__(self, qnode, tnode, score):
        self.qnode = qnode
        self.tnode = tnode
        self.score = score

    def is_leaf_pair(self):
        return self.qnode.CATEGORY == jstatutree.etypes.CATEGORY_TEXT and self.tnode.CATEGORY == jstatutree.etypes.CATEGORY_TEXT

    def to_tuple(self, node_as_code=False):
        return (self.qnode.code, self.tnode.code, self.score) if node_as_code else (self.qnode, self.tnode, self.score)

    def __str__(self):
        return "{0}-{1}: {2}".format(self.qnode.code, self.tnode.code, self.score)


class EdgeStore(object):
    def __init__(self):
        self.from_qcode = trie.StringTrie()
        self.from_tcode = trie.StringTrie()
        self.edges = []
        self.edge_counter = Counter()

    def add_edge(self, qnode: Element, tnode: Element, score: float):
        self.edges.append(Edge(qnode, tnode, score))
        if qnode.code in self.from_qcode:
            self.from_qcode[qnode.code].append(len(self) - 1)
        else:
            self.from_qcode[qnode.code] = [len(self) - 1]
        if tnode.code in self.from_tcode:
            self.from_tcode[tnode.code].append(len(self) - 1)
        else:
            self.from_tcode[tnode.code] = [len(self) - 1]
        self.edge_counter[tnode.code[:14]] += 1

    def __len__(self):
        return len(self.edges)

    def filtering(self, key):
        return [e for e in self.edges if key(e)]

    def iter_edges(self, *, qkey: str = None, tkey: str = None):
        logger = _get_logger('EdgeStore.iter_edges')
        qcode = qkey.code if isinstance(qkey, Element) else qkey
        tcode = tkey.code if isinstance(tkey, Element) else tkey
        if qcode and tcode:
            try:
                qedge_indices = set(i for l in self.from_qcode[qcode:] for i in l)
            except KeyError:
                logger.warning('KeyError has raised for qcode: %s', str(qcode))
                logger.debug('key list: \n %s', '\n'.join(self.from_qcode.keys()))
                qedge_indices = set()
                raise
            try:
                tedge_indices = set(i for l in self.from_tcode[tcode:] for i in l)
            except KeyError:
                logger.warning('KeyError has raised for tcode: %s', str(tcode))
                logger.debug('key list: \n %s', '\n'.join(self.from_tcode.keys()))
                tedge_indices = set()
                raise
            edge_indices = qedge_indices & tedge_indices
        elif qcode:
            try:
                edge_indices = set(i for l in self.from_qcode[qcode:] for i in l)
            except KeyError:
                logger.warning('KeyError has raised for qcode: %s', str(qcode))
                logger.debug('key list: \n %s', '\n'.join(self.from_qcode.keys()))
                edge_indices = set()
        elif tcode:
            try:
                edge_indices = set(i for l in self.from_tcode[tcode:] for i in l)
            except KeyError:
                logger.warning('KeyError has raised for tcode: %s', str(tcode))
                logger.debug('key list: \n %s', '\n'.join(self.from_tcode.keys()))
                edge_indices = set()
        else:
            edge_indices = range(len(self))
        logger.debug('iterator size: %d(/%d)', len(edge_indices), len(self))
        yield from (self.edges[i] for i in edge_indices)

    def iter_tree_codes(self):
        yield from self.edge_counter.keys()


    def extract_codes(func):
        def wrapper(self, item):
            if isinstance(item, Edge):
                qcode = item.qnode.code
                tcode = item.tnode.code
            else:
                qcode = item[0].code if isinstance(item[0], Element) else item[0]
                tcode = item[1].code if isinstance(item[1], Element) else item[1]
            return func(self, qcode, tcode)
        return wrapper

    @extract_codes
    def __contains__(self, qcode, tcode):
        return qcode in self.from_qcode and tcode in self.from_tcode and \
               len(set(self.from_qcode[qcode]) & set(self.from_tcode[tcode]))

    @extract_codes
    def find_edge(self, qcode, tcode):
        logger = _get_logger('EdgeStore.find_edge')
        idxs = list(set(self.from_qcode[qcode]) & set(self.from_tcode[tcode]))
        if len(idxs) > 1:
            logger.warning('Multiple edge found in a node pain.')
            for idx in idxs:
                logger.warning(str(self.edges[idx]))
        return self.edges[idxs[0]]

    @extract_codes
    def leaf_max_match(self, qcode, tcode):
        G = nx.Graph()
        logger = _get_logger('EdgeStore.leaf_max_match')
        edge_counter = 0
        for edge in self.iter_edges(tkey=tcode, qkey=qcode):
            if not edge.is_leaf_pair():
                continue
            G.add_node(edge.qnode.code, bipartite=0)
            G.add_node(edge.tnode.code, bipartite=1)
            G.add_edge(edge.qnode.code, edge.tnode.code, weight=edge.score)
            edge_counter += 1
        t = time.time()
        mm = nx.max_weight_matching(G)
        logger.info('MaxWeightMatch with %d edges finished in %f sec', edge_counter, time.time()-t)
        return [self.find_edge((q, t)) if (q, t) in self else self.find_edge((t, q))
                for q, t in mm]


class TraverserBase(abc.ABC):
    def __init__(self, query, target, edge_store: EdgeStore, *, logger=None):
        self.query = query
        self.target = target
        self.edge_queue = Queue()
        self.edge_store = edge_store
        for e in edge_store.iter_edges():
            self.edge_queue.put(e)

    @abc.abstractmethod
    def traverse(self):
        pass

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

    def _tree_name_factory(self, elem):
        if str(self.query_tree.lawdata.code) in str(elem.code):
            return self.query_tree.lawdata.name
        return self.match_tree.lawdata.name

    """
    def find_matching_edge(self, threshold: float, *, _target_node: Element = None) -> Generator[Edge, None, None]:
        target_node = _target_node or self.match_tree.getroot()
        for e in sorted(self.edge_store.iter_edges(tkey=target_node.code), key=lambda x: str(x).count('/')):
        if self.edge_store.find_edge((self.query_tree
                                              .lawdata.code, target_node.code)).score >= threshold:
            yield target_node
            return
        elif target_node.CATEGORY >= jstatutree.CATEGORY_TEXT:
            return
        for child in target_node:
            yield from self.find_matching_edge(threshold, _target_node=child)     
    """

    def find_matching_edge(self, threshold: float):
        candidates = self.edge_store.filtering(key=lambda x: x.score >= threshold)
        results = []
        for e in sorted(candidates, key=lambda e: str(e.qnode.code).count('/')**2+str(e.tnode.code).count('/')**2):
            for r in results:
                if r.qnode.code in e.qnode.code:
                    break
            else:
                results.append(e)
        return results

    def comp_table(self, threshold: float, **kwargs):
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
    class ActivatorBase(abc.ABC):
        def initial_edges(self, edges: list) -> EdgeStore:
            edge_store = EdgeStore()
            for e in edges:
                q, t, score = e.to_tuple()
                edge_store.add_edge(q, t, self.activate(score))
            return edge_store

        @abc.abstractmethod
        def activate(self, val: float) -> float:
            pass

    class Void(ActivatorBase):
        def activate(self, val: float) -> float:
            return val

    class Sigmoid(ActivatorBase):
        def __init__(self, gain: int, center: float):
            self.gain = gain
            self.center = center

        def activate(self, val: float) -> float:
            return 1 / (1 + math.exp((self.center - val) * self.gain))

class PolicyMatchFactory(object):
    def __init__(self, query_tree: Jstatutree, tree_factory):
        self.query_tree = query_tree
        self.self = EdgeStore()
        self.tree_store = {}
        self.tree_factory = tree_factory
        self.leaf_edge_store = EdgeStore()

    def matching(self, scorer, traverser_cls, activator, * , least_matching_node=0):
        logger = _get_logger('PolicyMatchFactory.matching')
        logger.info('matching() called')
        obj = PolicyMatchObject()
        for target_tree, matching_edge_store in self.scoring(scorer, traverser_cls, activator, least_matching_node=least_matching_node):
            obj.add_unit(self.query_tree, target_tree, matching_edge_store)
        logger.info('matching() finished')
        return obj

    def add_leaf(self, qcode: str, tcode: str, similarity: float):
        logger = _get_logger('PolicyMatchFactory.add_leaf')
        qnode = self.query_tree.find_by_code(qcode)
        rcode = tcode[:14]
        if rcode not in self.tree_store:
            self.tree_store[rcode] = self.tree_factory(rcode)
        tnode = self.tree_store[rcode].find_by_code(tcode)
        logger.debug("add edge %s %s %f", str(qnode.code), str(tnode.code), similarity)
        self.leaf_edge_store.add_edge(qnode, tnode, similarity)

    def scoring(self, traverser_cls, scorer, activator: SimilarityActivator.ActivatorBase, *, least_matching_node=0):
        logger = _get_logger('PolicyMatchFactory.scoring')
        for target_code, target_tree in self.tree_store.items():
            if least_matching_node > 0 and \
                    len(list(self.leaf_edge_store.iter_edges(tkey=target_code))) < least_matching_node:
                logger.info('skip(single node matching): %s', str(target_code))
                continue
            logger.info('scoring %s', target_tree.lawdata.code)
            target = target_tree.getroot()
            matching_edge_store = activator.initial_edges(list(self.leaf_edge_store.iter_edges(tkey=target.code)))
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
    class Traverser(TraverserBase):
        def get_target_parent(self, tnode: Element) -> Element:
            target_parent = self.target.find_by_code(str(Path(tnode.code).parent))
            if target_parent is not None and len(target_parent) == len(tnode):
                return self.get_target_parent(target_parent)
            return target_parent

        def get_query_parent(self, qnode: Element) -> Element:
            query_parent = self.query.find_by_code(str(Path(qnode.code).parent))
            if query_parent is not None and len(query_parent) == len(qnode):
                return self.get_query_parent(query_parent)
            return query_parent

        def traverse(self):
            logger = _get_logger('Al'
                                 'ternately.Traverser.traverse')
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
        class LinearCombination(object):
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

        class GeometricMean(object):
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

        class QueryLeafCoverage(object):
            def __init__(self, using_qweight=False):
                self.using_qweight = using_qweight

            def scoring(self, edge:Edge, edge_store: EdgeStore, **other_results):
                logger = _get_logger('Alternately.Scorer.QueryLeafCoverage.scoring')
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

        class TargetLeafCoverage(object):
            def scoring(self, edge:Edge, edge_store: EdgeStore, **other_results):
                logger = _get_logger('Alternately.Scorer.TargetLeafCoverage.scoring')
                tleaf_num = sum(1 for _ in edge.tnode.iterXsentence(include_code=False, include_value=False))
                matching = other_results.get('leaf_max_match', None) or edge_store.leaf_max_match(edge)
                other_results['leaf_max_match'] = matching
                match_num = len(matching)
                score = match_num/tleaf_num
                logger.debug('%d/%d = %f',match_num, tleaf_num, score)
                return score, other_results

        class QueryLeafProximity(object):
            def __init__(self, using_qweight=False):
                self.using_qweight = using_qweight

            def scoring(self, edge:Edge, edge_store: EdgeStore, **other_results):
                logger = _get_logger('Alternately.Scorer.QueryLeafCoverage.scoring')
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

        class TargetLeafProximity(object):
            def scoring(self, edge:Edge, edge_store: EdgeStore, **other_results):
                logger = _get_logger('Alternately.Scorer.QueryLeafCoverage.scoring')
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


