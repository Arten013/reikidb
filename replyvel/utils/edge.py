import networkx as nx
from networkx.algorithms import bipartite
import jstatutree
from jstatutree.element import Element
import pygtrie as trie
from .logger import get_logger
import time
from collections import Counter


class Edge(object):
    def __init__(self, qnode, tnode, score):
        self.qnode = qnode
        self.tnode = tnode
        self.score = score

    def is_leaf_pair(self):
        return self.qnode.CATEGORY == jstatutree.etypes.CATEGORY_TEXT and \
               self.tnode.CATEGORY == jstatutree.etypes.CATEGORY_TEXT

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
        logger = get_logger('EdgeStore.iter_edges')
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
        logger = get_logger('EdgeStore.find_edge')
        idxs = list(set(self.from_qcode[qcode]) & set(self.from_tcode[tcode]))
        if len(idxs) > 1:
            logger.warning('Multiple edge found in a node pain.')
            for idx in idxs:
                logger.warning(str(self.edges[idx]))
        return self.edges[idxs[0]]

    @extract_codes
    def leaf_max_match(self, qcode, tcode):
        G = nx.Graph()
        logger = get_logger('EdgeStore.leaf_max_match')
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
