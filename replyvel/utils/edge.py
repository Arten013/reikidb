import networkx as nx
from networkx.algorithms import bipartite
import jstatutree
from jstatutree import Jstatutree
from jstatutree.element import Element
import pygtrie as trie
from .logger import get_logger
import time
from collections import Counter
import numpy as np
from typing import Sequence, Union, Any, NewType, Callable, TypeVar
import copy
import re

EdgeScoreType = NewType('EdgeScoreType', Union[float, np.ndarray])
X = TypeVar('X')

class Edge(object):
    def __init__(self, qnode: Element, tnode: Element, score: float = np.NaN):
        self.qnode = qnode
        self.tnode = tnode
        self.score = score if score is not None else np.NaN
        assert isinstance(score, float)
        self.gviz_attrib = {'label': ''}
        self.is_aligner = False

    def is_leaf_pair(self):
        return self.qnode.CATEGORY == jstatutree.etypes.CATEGORY_TEXT and \
               self.tnode.CATEGORY == jstatutree.etypes.CATEGORY_TEXT

    def to_tuple(self, node_as_code=False):
        return (self.qnode.code, self.tnode.code, self.score) if node_as_code else (self.qnode, self.tnode, self.score)

    def is_scored(self):
        return not np.isnan(self.score)

    def __str__(self):
        if not self.is_scored():
            return "{0}-{1}".format(self.qnode.code, self.tnode.code)
            #return "{0}-{1}: {2}".format(self.qnode.code, self.tnode.code, self.score)
        else:
            return "{0}-{1}: {2}".format(self.qnode.code, self.tnode.code, self.score)

    def add_to_nxbipartite(self, G: nx.DiGraph, *args, **kwargs) -> nx.DiGraph:
        G.add_node(self.qnode.code, bipartite=0)
        G.add_node(self.tnode.code, bipartite=1)
        G.add_edge(self.qnode.code, self.tnode.code, weight=self.score)
        return G


EdgeLikeType = NewType('EdgeLikeType', Union[Edge, Sequence[str], Sequence[Element]])


def edge_like_to_codes(item: EdgeLikeType):
    qcode, tcode = None, None
    if isinstance(item, Edge):
        qcode = item.qnode.code
        tcode = item.tnode.code
    elif isinstance(item, tuple):
        if len(item) >= 2 and isinstance(item[0], str) and isinstance(item[1], str):
            qcode, tcode = item[:2]
        elif len(item) >= 2 and isinstance(item[0], str) and isinstance(item[1], str):
            qcode, tcode = item[0].code, item[1].code
    if qcode is None:
        get_logger('_edge_like_to_codes').warning('Non edge-like instance passed ({}).'.format(repr(item)))
    return str(qcode), str(tcode)

class ParallelizedEdge(Edge):
    def __init__(self, qnode: Element, tnode: Element, score: np.ndarray):
        self.qnode = qnode
        self.tnode = tnode
        self.score = score
        assert score is not None, 'Pass NaN array for non-scored edge.'
        self.gviz_attrib = {'label': ''}
        self.is_aligner = False

    def is_scored(self):
        return sum(np.isnan(self.score)) != self.score.shape[0]

    @classmethod
    def from_edge(cls, edge: Edge, new_score: np.ndarray):
        return cls(edge.qnode, edge.tnode, new_score)

    def add_to_nxbipartite(self, G: nx.DiGraph, score_idx: int = 0) -> nx.DiGraph:
        G.add_node(self.qnode.code, bipartite=0)
        G.add_node(self.tnode.code, bipartite=1)
        G.add_edge(self.qnode.code, self.tnode.code, weight=self.score[score_idx])
        return G


class EdgeStoreCore(object):
    def __init__(self):
        self.from_qcode = trie.StringTrie()
        self.from_tcode = trie.StringTrie()
        self._edges = []
        self.edge_count_per_tree = Counter()

    @property
    def edges(self):
        raise Exception('Do not operate EdgeStore.edges directly.')

    def _check_edge_idx(self, idx: int) -> bool:
        return self._edges[idx] is not None

    def _iter_idx_from_code(self, code: str, is_qcode=True):
        gen = self.from_qcode[code:] if is_qcode else self.from_tcode[code:]
        try:
            return set(i for l in gen for i in l)
        except KeyError:
            logger = get_logger('EdgeStore._iter_idx_from_code')
            if is_qcode:
                logger.debug('KeyError has raised for qcode: %s', str(code))
                logger.debug('key list: \n %s', '\n'.join(self.from_qcode.keys()))
            else:
                logger.debug('KeyError has raised for tcode: %s', str(code))
                logger.debug('key list: \n %s', '\n'.join(self.from_tcode.keys()))
            return set()

    def _find_valid_edge_indices(self, qcode: str, tcode: str):
        return list(
            set(i for i in self.from_qcode.get(qcode, []) if self._check_edge_idx(i))
            & set(i for i in self.from_tcode.get(tcode, []) if self._check_edge_idx(i))
        )

    def _copy_data(self, src):
        assert isinstance(src, EdgeStoreCore), 'Invalid Argument {}'.format(repr(src))
        self.from_tcode = copy.deepcopy(src.from_tcode)
        self.from_qcode = copy.deepcopy(src.from_qcode)
        self._edges = copy.deepcopy(src._edges)
        return

    def _delete_edge(self, idx: int):
        self._edges[idx] = None

    def delete_unscored_items(self):
        for i, e in enumerate(self._edges):
            if e is not None and not e.is_scored():
                self._edges[i] = None

    def delete_edge(self, item: EdgeLikeType):
        qcode, tcode = edge_like_to_codes(item)
        for i in self._find_valid_edge_indices(qcode, tcode):
            self._delete_edge(i)

    def add(self, qnode: Element, tnode: Element, score: EdgeScoreType, tag: str = None):
        if tag is not None and tag != 'default':
            get_logger('EdgeStore.add').warning('Any tag should not be specified in Non-parallelized EdgeStore.')
        if not isinstance(score, float):
            raise ValueError('non-float score used in Non-parallelized EdgeStore.')
        edge = Edge(qnode, tnode, score)
        self._edges.append(edge)
        if qnode.code in self.from_qcode:
            self.from_qcode[qnode.code].append(len(self) - 1)
        else:
            self.from_qcode[qnode.code] = [len(self) - 1]
        if tnode.code in self.from_tcode:
            self.from_tcode[tnode.code].append(len(self) - 1)
        else:
            self.from_tcode[tnode.code] = [len(self) - 1]
        self.edge_count_per_tree[tnode.code[:14]] += 1
        return edge

    def is_lca_pair_edge(self, edge: Edge) -> bool:
        valid_qkeys = set(k for k, ids in self.from_qcode.items() if sum(1 for i in ids if self._edges[i] is not None) > 0)
        valid_tkeys = set(k for k, ids in self.from_tcode.items() if sum(1 for i in ids if self._edges[i] is not None) > 0)
        retrieved_tkeys = set(e.tnode.code for e in self.iter_edges(qkey=edge.qnode.code))
        if valid_tkeys != retrieved_tkeys:
            return False
        retrieved_qkeys = set(e.qnode.code for e in self.iter_edges(tkey=edge.tnode.code))
        if valid_qkeys != retrieved_qkeys:
            return False
        return True

    def iter_edges(self, *, qkey: str = None, tkey: str = None):
        qcode = qkey.code if isinstance(qkey, Element) else qkey
        tcode = tkey.code if isinstance(tkey, Element) else tkey
        if qcode is None and tcode is None:
            yield from [e for i, e in enumerate(self._edges) if self._check_edge_idx(i)] #todo: list -> generator
        else:
            qedge_indices, tedge_indices = set(), set()
            if qcode is not None:
                qedge_indices = self._iter_idx_from_code(qcode, is_qcode=True)
            if tcode is not None:
                tedge_indices = self._iter_idx_from_code(tcode, is_qcode=False)
            if qcode is not None and tcode is not None:
                edge_indices = qedge_indices & tedge_indices
            else:
                edge_indices = qedge_indices | tedge_indices
            yield from (self._edges[i] for i in edge_indices if self._check_edge_idx(i))

    def get_edge(self, item: EdgeLikeType, default: X = None) -> Union[Edge, X]:
        qcode, tcode = edge_like_to_codes(item)
        indices = self._find_valid_edge_indices(qcode, tcode)
        if len(indices) > 1:
            logger = get_logger('EdgeStore.find_edge')
            logger.warning('Multiple edge found in a node pain.')
            for idx in indices:
                logger.warning(str(self._edges[idx]))
        elif len(indices) == 0:
            return default
        return self._edges[indices[0]]

    def __len__(self) -> int:
        return sum([1 for _ in self.iter_edges()])

    def __contains__(self, item: EdgeLikeType) -> bool:
        qcode, tcode = edge_like_to_codes(item)
        return len(self._find_valid_edge_indices(qcode, tcode)) > 0


class EdgeStore(EdgeStoreCore):
    def __init__(self):
        super().__init__()
        self._score_matrix = None
        self._matching_cache = {}

    """
    @property
    def score_matrix(self):
        if not self._score_matrix:
            self.reset_score_matrix()
        return self._score_matrix

    def reset_score_matrix(self):
        self._score_matrix = np.matrix([e.score for e in self.iter_edges()])
    """

    def add_edge(self, edge: Edge, tag: str = None):
        new_edge = self.add(*edge.to_tuple(), tag=tag)
        new_edge.gviz_attrib = edge.gviz_attrib
        new_edge.is_aligner = edge.is_aligner
        return new_edge

    def filtering(self, key):
        return [e for e in self.iter_edges() if key(e)]

    def iter_tree_codes(self):
        yield from self.edge_count_per_tree.keys()

    def find_edge(self, item: EdgeLikeType) -> Edge:
        ret = self.get_edge(item, default=None)
        if ret is None:
            raise KeyError(repr(item))
        return ret

    def leaf_max_match(self, item:EdgeLikeType) -> Sequence[Edge]:
        qcode, tcode = edge_like_to_codes(item)
        G = nx.Graph()
        logger = get_logger('EdgeStore.leaf_max_match')
        edge_counter = 0
        for edge in self.iter_edges(tkey=tcode, qkey=qcode):
            if not edge.is_leaf_pair():
                continue
            edge.add_to_nxbipartite(G, score_idx=0)
            edge_counter += 1
        t = time.time()
        mm = nx.max_weight_matching(G)
        logger.info('MaxWeightMatch with %d edges finished in %f sec', edge_counter, time.time() - t)
        return [self.find_edge((q, t)) if (q, t) in self else self.find_edge((t, q))
                for q, t in mm]

    # @property
    # def _lgmm_cache(self):
    #     if 'lgmm' not in self._matching_cache:
    #         self._matching_cache['lgmm'] = {}
    #     return self._matching_cache['lgmm']
    #
    # def leaf_greedy_modify_matching(self, item):
    #     edge = self.find_edge(item)
    #     res = self._lgmm_cache.get(str(edge))
    #     if res is not None:
    #         return res
    #     for e in self.iter_edges(qkey=edge.qnode.code,tkey=edge.tnode.code):
    #
    #     child_edges = [
    #         self.leaf_greedy_modify_matching((qcode, tcode))
    #         for
    #     ]

    def keys(self):
        yield 'default'

    def values(self):
        return self

    def items(self):  # for compatibility to parallelized version
        yield 'default', self


class ParallelizedEdgeStore(EdgeStore):
    def __init__(self, scorer_tags: Sequence[str]):
        super(ParallelizedEdgeStore, self).__init__()
        self._scorer_tags = scorer_tags
        self.tag_indices = {tag: i for i, tag in enumerate(self.scorer_tags)}

    # core implementation (operation with self._edge )#
    @classmethod
    def copy_from_edge_store(cls, src: EdgeStore, scorer_tags: Sequence[str], *, score_factory=None):
        score_factory = score_factory or (lambda f: np.array([f] * len(scorer_tags)))
        tar = cls(scorer_tags)
        tar.from_qcode = src.from_qcode
        tar.from_tcode = src.from_tcode
        tar._edges = [ParallelizedEdge.from_edge(e, new_score=score_factory(e.score)) for e in src._edges]
        return tar

    def add(self, qnode: Element, tnode: Element, score: EdgeScoreType, tag: str = None):
        assert (tag is None) or (tag in self.scorer_tags), 'Invalid tag {} (Not registered)'.format(tag)
        edge = self.get_edge((qnode.code, tnode.code))
        if edge is not None:
            if not edge.is_scored():
                score = np.zeros(len(self.scorer_tags))
            edge.score[self.tag_indices[tag]] = score
        else:
            if isinstance(score, np.ndarray):
                scores = score
            else:
                scores = np.zeros(len(self.scorer_tags))
                scores[self.tag_indices[tag]] = score
            edge = ParallelizedEdge(qnode, tnode, scores)
            self._edges.append(edge)
            if qnode.code in self.from_qcode:
                self.from_qcode[qnode.code].append(len(self) - 1)
            else:
                self.from_qcode[qnode.code] = [len(self) - 1]
            if tnode.code in self.from_tcode:
                self.from_tcode[tnode.code].append(len(self) - 1)
            else:
                self.from_tcode[tnode.code] = [len(self) - 1]
            self.edge_count_per_tree[tnode.code[:14]] += 1
        return edge

    # other implementation #
    @property
    def scorer_tags(self):
        return self._scorer_tags

    @scorer_tags.setter
    def scorer_tags(self, value: Any):
        raise Exception('You must not change scorer_tags directly.')

    def keys(self):
        yield from self.scorer_tags

    def values(self):
        yield from (es for t, es in self.items())

    def items(self, filtering_unscored_items=True):
        for i, tag in enumerate(self.scorer_tags):
            es = EdgeStore()
            es.from_tcode = copy.deepcopy(self.from_tcode)
            es.from_qcode = copy.deepcopy(self.from_qcode)
            es._edges = [Edge(e.qnode, e.tnode, e.score[i]) if e is not None else None
                         for e in self._edges]
            if filtering_unscored_items:
                es.delete_unscored_items()
            yield tag, es
