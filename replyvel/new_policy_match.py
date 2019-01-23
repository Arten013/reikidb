from jstatutree.element import Element
from jstatutree import Jstatutree, etypes
from typing import Generator, Mapping, Union, Sequence, Tuple
import math
import numpy as np
from collections import OrderedDict
from .utils.npm_abstracts import AbstractScorer, AbstractActivator
from .utils.edge import Edge, EdgeStore, ParallelizedEdgeStore
from .utils.logger import get_logger
from graphviz import Digraph
from collections import defaultdict

DefaultEdgeColorset = ['forestgreen', 'brown', 'deeppink1', 'cyan', 'gold2', 'crimson']


class ParallelScorer(AbstractScorer):
    def __init__(self):
        self.scorers = OrderedDict()

    @property
    def size(self):
        return len(self.scorers)

    @property
    def is_parallel(self):
        return True

    def reformatting_leaf_edges(self, *edges: Edge) -> Sequence[Edge]:
        for e in edges:
            if not isinstance(e, np.ndarray):
                e.score = np.array([e.score] * self.size)
        return edges

    @property
    def tags(self):
        return list(self.scorers.keys())

    def add(self, tag: str, scorer):
        self.scorers[tag] = scorer

    def add_by_keyvalue(self, keyvalue: Mapping):
        self.scorers.update(keyvalue)

    def scoring(self, edge: Edge, edge_store: EdgeStore, **other_results):
        score_list = []
        edge_store_dict = {k:v for k, v in edge_store.items()}
        for tag, scorer in self.scorers.items():
            single_score, single_other_results = scorer.scoring(edge, edge_store_dict[tag], **other_results)
            score_list += [single_score]
            other_results.update(single_other_results)
        return np.array(score_list), other_results


class PolicyMatchObject(object):
    def __init__(self, scorer_tags: Sequence[str]):
        self.units = {}
        self.scorer_tags = scorer_tags

    def add_unit(self, query_tree: Jstatutree, match_tree: Jstatutree, edge_store: EdgeStore):
        self.units[str(match_tree.lawdata.code)] = PolicyMatchUnit(query_tree, match_tree, edge_store)

    def ranking(self, threshold: float, topn: int=3, main_tag: str=None):
        main_tag_idx = 0 if main_tag is None else [i for i, t in enumerate(self.scorer_tags) if t == main_tag][0]
        results = [(k, list(u.find_matching_edge(threshold))) for k, u in self.units.items()]
        def rerank(stag: str, results: Sequence[Edge], edge_store: EdgeStore):
            penalty = 100000000
            for e in results:
                penalty = min(max(e.qnode.LEVEL, e.tnode.LEVEL), penalty)
            return penalty
        return sorted(results, key=lambda x: rerank(*x[1][main_tag_idx]))[:topn]

    def comp_table(self, threshold: float, topn: int=3, *args, **kwargs):
        for k, u in self.ranking(threshold, topn):
            yield self.units[k].comp_table(threshold, *args, **kwargs)


class PolicyMatchUnit(object):
    def __init__(self, query_tree: Jstatutree, match_tree: Jstatutree, edge_store: EdgeStore):
        self.query_tree = query_tree
        self.match_tree = match_tree
        self.edge_store = edge_store

    def find_matching_edge(self, threshold: float, query_root: str=None, match_root: str=None) -> list:
        for stag, edge_store in self.edge_store.items():
            candidates = edge_store.filtering(key=lambda x: x.score >= threshold
                                                            and not x.is_leaf_pair()
                                                            #and x.qnode.CATEGORY < etypes.CATEGORY_TEXT
                                                            #and x.tnode.CATEGORY < etypes.CATEGORY_TEXT
                                                            and (query_root is None or query_root in str(x.qnode.code))
                                                            and (match_root is None or match_root in str(x.tnode.code))
                                              )
            def sortkey(e):
                q = str(e.qnode.code).count('/')
                t = str(e.tnode.code).count('/')
                if q > t:
                    return q, -e.score
                    #return q, q-t, -e.score
                else:
                    return t, -e.score
                    #return t, t-q, -e.score
            yield stag, sorted(candidates, key=sortkey), edge_store

    def comp_table(self, threshold: float, *, edge_colorset: Mapping = None, sub_branch: str='ALL', query_root: str=None, match_root: str=None, **kwargs) -> Digraph:
        graph_edge_store = ParallelizedEdgeStore(scorer_tags=list(self.edge_store.keys()))
        edge_colorset = edge_colorset or {}
        for i, (tag, edges, edge_store) in enumerate(self.find_matching_edge(threshold, query_root, match_root)):
            for e in edges:
                e = self.edge_store.find_edge(e)
                if e in graph_edge_store:
                    e.gviz_attrib['xlabel'] = tag+ '・' + e.gviz_attrib['label']
                    continue
                e = graph_edge_store.add_edge(e)
                e.gviz_attrib['xlabel'] = '{}({})'.format(
                    tag,
                    ', '.join(
                        '{0}:{1}'.format(k, round(e.score[j], 3)) for j, k in enumerate(self.edge_store.keys())
                    )
                )
                e.gviz_attrib['color'] = edge_colorset.get(tag, DefaultEdgeColorset[i])
                e.gviz_attrib['style'] = 'solid'
                e.gviz_attrib['tailport'] = 'e'
                e.gviz_attrib['headport'] = 'w'
                if not sub_branch:
                    continue
                elif sub_branch == 'maxmatch':
                    edge_gen = edge_store.leaf_max_match(e)
                else:
                    edge_gen = edge_store.iter_edges(qkey=str(e.qnode.code), tkey=str(e.tnode.code))
                for me in edge_gen:
                    if me in graph_edge_store or not me.is_leaf_pair():
                        continue
                    me = graph_edge_store.add_edge(self.edge_store.find_edge(me))
                    me.gviz_attrib['xlabel'] = '{0}'.format(round(me.score[0], 3))
                    # me.gviz_attrib['color'] = edge_colorset.get(tag, DefaultEdgeColorset[i])
                    me.gviz_attrib['style'] = 'dotted'
                    me.gviz_attrib['tailport'] = 'e'
                    me.gviz_attrib['headport'] = 'w'
                    me.is_aligner = True
        factory = CompTableGraphFactory()
        return factory.construct(
            self.query_tree.change_root(query_root) if query_root else self.query_tree,
            self.match_tree.change_root(match_root) if match_root else self.match_tree,
            graph_edge_store
        )

    def unit_comp_tables(self, threshold: float, *, sub_branch: str='maxmatch', query_root: str=None, match_root: str=None, **kwargs) -> Digraph:
        for i, (tag, edges, edge_store) in enumerate(self.find_matching_edge(threshold, query_root, match_root)):
            for e in edges:
                graph_edge_store = EdgeStore()
                e = graph_edge_store.add_edge(e)
                e.gviz_attrib['xlabel'] = '{0}:{1}'.format(tag, round(e.score, 3))
                e.gviz_attrib['style'] = 'solid'
                e.gviz_attrib['tailport'] = 'e'
                e.gviz_attrib['headport'] = 'w'
                if not sub_branch:
                    continue
                elif sub_branch == 'maxmatch':
                    edge_gen = edge_store.leaf_max_match(e)
                else:
                    edge_gen = edge_store.iter_edges(qkey=str(e.qnode.code), tkey=str(e.tnode.code))
                for me in edge_gen:
                    if me in graph_edge_store or not me.is_leaf_pair():
                        continue
                    me = graph_edge_store.add_edge(me)
                    me.gviz_attrib['xlabel'] = '{0}:{1}'.format('Similarity', round(me.score, 3))
                    me.gviz_attrib['style'] = 'dotted'
                    me.gviz_attrib['tailport'] = 'e'
                    me.gviz_attrib['headport'] = 'w'
                    me.is_aligner = True
                factory = CompTableGraphFactory()
                yield tag, e, factory.construct(
                    self.query_tree.change_root(e.qnode.code),
                    self.match_tree.change_root(e.tnode.code),
                    graph_edge_store
                )


class CompTableGraphFactory(object):
    def construct(self, query_tree: Jstatutree, match_tree: Jstatutree, edges: EdgeStore) -> Digraph:
        G = Digraph('comptable', filename="hoge")
        G.body.append('\tgraph [ newrank=true compound=true; ];')
        G.body.append('\tsplines=false;')
        G.body.append('\trankdir=TB;')
        G.body.append('\tnodesep=2;')
        G.body.append('\tlayout="dot";')
        G.node('QStart', label='QStart', style='invis', weight='100')
        G.node('QEnd', label='QEnd', style='invis', weight='100')
        G.node('MStart', label='MStart', style='invis', weight='100')
        G.node('MEnd', label='MEnd', style='invis', weight='100')
        G.edge('QStart', 'MStart', style='invis', weight='100')
        G.body.append('{rank=same; QStart MStart;}')
        G.edge('QEnd', 'MEnd', style='invis', weight='100')
        G.body.append('{rank=same; QEnd MEnd;}')
        G.node('QStart', label='QEnd', style='invis', weight='100')
        G.node('MStart', label='MEnd', style='invis', weight='100')

        colorset = defaultdict(lambda: 'black')
        colorset.update(
            {
                'Article': 'red',
                'Paragraph': 'darkorchid',
                'Item': 'darkorchid',
                'Sentence': 'blue'
            }
        )
        query_graph = self.get_graph(query_tree.getroot(), query_tree.lawdata.name, colorset)
        target_graph = self.get_graph(match_tree.getroot(), match_tree.lawdata.name, colorset)

        G.subgraph(query_graph['graph'])
        G.edge('QStart', query_graph['head'].code, style='invis', weight='1000')
        G.edge(query_graph['tail'].code, 'QEnd', style='invis', weight='1000')
        G.subgraph(target_graph['graph'])
        G.edge('MStart', target_graph['head'].code, style='invis', weight='1000')
        G.edge(target_graph['tail'].code, 'MEnd', style='invis', weight='1000')

        for edge in edges.iter_edges(qkey=str(query_tree.lawdata.code), tkey=str(match_tree.lawdata.code)):
            qleaf = self.get_sample_leaf(edge.qnode)
            tleaf = self.get_sample_leaf(edge.tnode)
            G.edge(
                qleaf,
                tleaf,
                ltail=("cluster_" + edge.qnode.code) if qleaf != edge.qnode.code else None,
                lhead=("cluster_" + edge.tnode.code) if tleaf != edge.tnode.code else None,
                **edge.gviz_attrib
            )

        max_rank = -1
        for ql in query_graph['leaves']:
            #print(ql.code)
            #if not edges.from_qcode.has_subtrie(str(ql.code)):
            #    continue
            tedges = list(e for e in edges.iter_edges(qkey=ql.code) if e.is_aligner)
            #print(tedges)
            if len(tedges) == 0:
                continue
            tcode = tedges[0].tnode.code
            for i, tl in enumerate(target_graph['leaves'][max_rank + 1:]):
                if tl.code == tcode:
                    # print('{rank=same;' + '"' + ql.code + '"' + '; ' + '"' + tl.code + '"' + ';}')
                    max_rank = i + max_rank + 1
                    #G.body.append('{rank=same;' + '"' + ql.code + '"' + '; ' + '"' + tl.code + '"' + ';}')
                    break
        return G

    def get_graph(self, elem, lawname, colorset, leaf_edges=True):
        item = elem.to_dot(lawname=lawname, return_sides=True, colorset=colorset, return_clusters=True,
                           return_leaves=True, leaf_edges=leaf_edges)
        assert len(item) == 5, str([type(v) for v in item])
        return {k: v for k, v in zip(['graph', 'head', 'tail', "clusters", "leaves"], item)}

    @staticmethod
    def get_sample_leaf(node: Element):
        leaves = list(node.iterXsentence_code())
        return leaves[len(leaves) // 2]


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

    def matching(self, scorer, traverser_cls, activator, *,threshold:float=None, least_matching_node=0, target_codes=None):
        logger = get_logger('PolicyMatchFactory.matching')
        logger.info('matching() called')
        obj = PolicyMatchObject(scorer.tags)
        if threshold is None:
            for target_tree, matching_edge_store in self.scoring_all(traverser_cls, scorer, activator,
                                                             least_matching_node=least_matching_node):
                obj.add_unit(self.query_tree, target_tree, matching_edge_store)
        else:
            for target_tree, matching_edge_store in self.scoring(traverser_cls, scorer, activator,
                                                                    threshold=threshold,
                                                                     least_matching_node=least_matching_node,
                                                                    target_codes=target_codes
                                                                 ):
                obj.add_unit(self.query_tree, target_tree, matching_edge_store)
        logger.info('matching() finished')
        return obj

    def add_leaf_by_nodes(self, qnode: Element, tcode: str, similarity: float):
        logger = get_logger('PolicyMatchFactory.add_leaf')
        rcode = tcode[:14]
        if rcode not in self.tree_store:
            self.tree_store[rcode] = self.tree_factory(rcode)
        tnode = self.tree_store[rcode].find_by_code(tcode)
        logger.debug("add edge %s %s %f", str(qnode.code), str(tnode.code), similarity)
        self.leaf_edge_store.add(qnode, tnode, similarity)

    def add_leaf(self, qcode: str, tcode: str, similarity: float):
        logger = get_logger('PolicyMatchFactory.add_leaf')
        qnode = self.query_tree.find_by_code(qcode)
        rcode = tcode[:14]
        if rcode not in self.tree_store:
            self.tree_store[rcode] = self.tree_factory(rcode)
        tnode = self.tree_store[rcode].find_by_code(tcode)
        logger.debug("add edge %s %s %f", str(qnode.code), str(tnode.code), similarity)
        self.leaf_edge_store.add(qnode, tnode, similarity)

    def scoring_all(self, traverser_cls, scorer: AbstractScorer, activator: AbstractActivator, *, least_matching_node=0):
        logger = get_logger('PolicyMatchFactory.scoring') # todo: drop children
        for target_code, target_tree in self.tree_store.items():
            if least_matching_node > 0 and \
                    len(list(self.leaf_edge_store.iter_edges(tkey=target_code))) < least_matching_node:
                logger.debug('skip(single node matching): %s', str(target_code))
                continue
            logger.info('scoring %s', target_tree.lawdata.code)
            target = target_tree.getroot()
            edges = list(self.leaf_edge_store.iter_edges(tkey=target.code))
            # reformatted_edges = scorer.reformatting_leaf_edges(*edges)
            matching_edge_store = activator.initial_edges(edges)
            if isinstance(scorer, ParallelScorer):
                matching_edge_store = ParallelizedEdgeStore.copy_from_edge_store(matching_edge_store,
                                                                                 list(scorer.scorers.keys()))
            logger.debug('initial edge store size: %d', len(matching_edge_store))
            traverser = traverser_cls(self.query_tree.getroot(), target, matching_edge_store).traverse()
            edge = next(traverser)
            scoring_count = 0
            while True:
                if edge is None:
                    break
                scoring_count += 1
                edge.score = scorer.scoring(edge, matching_edge_store)[0]
                matching_edge_store.add(*edge.to_tuple())
                logger.debug('add edge: %s', str(matching_edge_store.find_edge(edge)))
                print('add edge: %s', str(matching_edge_store.find_edge(edge)))
                edge = traverser.send([edge.score, 1.0, 0.0])
            logger.info('finished scoring %s', target_tree.lawdata.code)
            yield target_tree, matching_edge_store
            print(target_code, scoring_count)
        logger.debug('quit')

    def scoring(self, traverser_cls, scorer: AbstractScorer, activator: AbstractActivator, threshold:float, *, least_matching_node=0, target_codes=None):
        assert not scorer.is_parallel, "you cannot use parallel scorer for scoring"
        logger = get_logger('PolicyMatchFactory.scoring') # todo: drop children
        scorer.reset_cache()
        entire_scoring_count = 0
        entire_edge_count = 0
        for target_code, target_tree in self.tree_store.items():
            if str(target_codes) is not None and target_code not in target_codes:
                continue
            print(target_code)
            scorer.reset_tmp_cache()
            if least_matching_node > 0 and \
                    len(list(self.leaf_edge_store.iter_edges(tkey=target_code))) < least_matching_node:
                logger.debug('skip(single node matching): %s', str(target_code))
                continue
            logger.info('scoring %s', target_tree.lawdata.code)
            target = target_tree.getroot()
            edges = list(self.leaf_edge_store.iter_edges(tkey=target.code))
            matching_edge_store = activator.initial_edges(edges)
            logger.debug('initial edge store size: %d', len(matching_edge_store))
            traverser = traverser_cls(self.query_tree.getroot(), target, matching_edge_store, threshold).traverse()
            edge = next(traverser)
            matching_edge_store.init_marker('delete')
            matching_edge_store.init_marker('used_leaf')
            missing_link = 0
            scoring_count = 0
            edge_count = 0
            while True:
                if edge is None:
                    break
                if edge is False:
                    edge = traverser.send([1, scorer, matching_edge_store, missing_link])
                    continue
                scoring_count += 1
                edge.score = scorer.scoring(edge, matching_edge_store)[0]
                matching_edge_store.add(*edge.to_tuple())
                if edge.score >= threshold:
                    edge_count += 1
                    #matching_edge_store.delete_descendant_edges(edge, delete_leaves=False)
                    logger.debug('add edge: %s', str(matching_edge_store.find_edge(edge)))
                    edge = traverser.send([edge.score, scorer, matching_edge_store, missing_link])
                else:
                    matching_edge_store.mark_edge('delete', edge)
                    edge = traverser.send([edge.score, scorer, matching_edge_store, missing_link])
            logger.info('finished scoring %s', target_tree.lawdata.code)

            matching_edge_store.delete_marked_edge('delete')
            for e in matching_edge_store.iter_edges():
                if e.is_leaf_pair():
                    continue
                if e.score < threshold:
                    matching_edge_store.delete_edge(e)
            for e in matching_edge_store.iter_edges():
                if not e.is_leaf_pair() and matching_edge_store.has_ancestor_pair(e, include_self=False):
                    matching_edge_store.mark_edge('delete', e)
            matching_edge_store.delete_marked_edge('delete')
            yield target_tree, matching_edge_store
            #print(target_code, scoring_count)
            entire_scoring_count += scoring_count
            entire_edge_count += edge_count
        print('entire scoring:', entire_scoring_count)
        print('edge count:', entire_edge_count)
        logger.debug('quit')