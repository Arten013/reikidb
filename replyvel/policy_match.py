import re
import math
from pathlib import Path
from itertools import groupby
from collections import Counter
import numpy as np
import jstatutree
import dill
import concurrent
from copy import copy



def lca(*keys):
    #print(*keys)
    if len(keys) == 0:
        raise ValueError("You must pass at least one argument.")
    lca = Path("")
    for part in Path(keys[0]).parts:
        next_lca = lca/part
        if sum((str(next_lca) in k) for k in keys[1:]) < len(keys[1:]):
            break
        lca = next_lca
    return str(lca)

class PolicyMatchObject(object):
    def __init__(self):
        self.queries = {}
        self.query_tree = None
        self.match_trees = {}
    
    def summary(self):
        return re.sub(r'\n +', '\n', """ ** policy match object info **
        query:
        {query}
        candidate trees: {mtree_count}""".format(
            query='\n\t'.join(q for q in self.queries.keys()),
            mtree_count=len(self.match_trees)
                  ))
    
    def calc_policy_forest(self, threshold, *, topn=5, prox_method='max', return_diff=False):
        forest = []
        for tree in self.match_trees.values():
            forest.extend(self.get_match_elements(tree.getroot(), threshold, prox_method=prox_method, return_diff=return_diff))
        return sorted(forest, key=lambda x: -sum(s for qt, rt, s in x[1]))[:topn]
    
    def _tree_name_factory(self, elem):
        key = str(jstatutree.lawdata.ReikiCode(elem.code))
        if key == str(self.query_tree.lawdata.code):
            return self.query_tree.lawdata.name
        return self.match_trees[key].lawdata.name
    
    def find_elem(self, key):
        rcode = str(jstatutree.lawdata.ReikiCode(key))
        if str(self.query_tree.lawdata.code) in rcode:
            return self.query_tree.find_by_code(key)
        tree = self.match_trees.get(rcode)
        if tree is None:
            raise ValueError('Invalid argument '+str(key))
        #print(rcode, key)
        return tree.find_by_code(key)
    
    def bind_by_lca(self, *keys):
        keys = set(keys)
        if len(keys) == 1:
            elem = self.find_elem(list(keys)[0])
            assert elem is not None,list(keys)[0]
            return elem.copy()
        lca_code = lca(*keys)
        lca_etype = jstatutree.etypes.code2etype(lca_code)
        elem = self.find_elem(lca_code)
        assert elem is not None, str(lca_code, self.match_trees)
        elem = elem.copy()
        children = []
        gchildren = {}
        for c_code in [c.code for c in elem]:
            for key in keys:
                if c_code == key:
                    children.append(self.find_elem(key))
                elif c_code in key:
                    gchildren[c_code] = gchildren.get(c_code, [c_code])
                    gchildren[c_code].append(key)
                    c_code not in children and children.append(c_code)
                else:
                    pass
        assert len(children) > 0
        elem._children = []
        for c in children:
            if isinstance(c, str):
                elem._children.append(self.bind_by_lca(*gchildren[c])) 
            else:
                elem._children.append(c)
        return elem
    
    def get_submatch(self, telement, minimum_query=True):
        if isinstance(telement, str):
            telement = self.find_elem(telement)
        elif isinstance(telement, jstatutree.Jstatutree):
            telement = telement.getroot()
        elif isinstance(telement, jstatutree.element.Element):
            telement = telement
        else:
            raise ValueError("Invalid Argument: {}".format(telement))
        sub_match_obj = PolicyMatchObject()
        if minimum_query:
            if len(telement.attrib.get("score", {})) > 0:
                qcode = lca(* telement.attrib["score"].keys())
                sub_match_obj.queries = {qcode:self.find_elem(qcode)}
            else:
                sub_match_obj.queries = {}
        else:
            sub_match_obj.queries = self.queries
        sub_match_obj.query_tree = self.query_tree
        key = str(jstatutree.lawdata.ReikiCode(telement.code))
        match_tree = copy(self.match_trees[key])
        match_tree._root = telement
        sub_match_obj.match_trees = {key: match_tree}
        return sub_match_obj
    
    def split_into_submatches(self, threshold, *, prox_method="max", topn=10):
        for telement, _, _ in self.calc_policy_forest(threshold, prox_method=prox_method, topn=topn):
            yield self.get_submatch(telement)
    
    def comp_table(self, threshold, prox_method="max", layout='dot', topn=5, **kwargs):
        forest = self.calc_policy_forest(threshold, prox_method=prox_method, topn=topn)
        if len(forest) == 0:
            print('Warning: comptable is empty')
            return jstatutree.graph.cptable_graph(
                self.query_tree.getroot(), 
                [tree.getroot() for tree in self.match_trees.values()], 
                [], [], tree_name_factory=self._tree_name_factory, layout=layout, **kwargs)
        telems = [
            self.bind_by_lca(*sub_keys)
            for rc, sub_keys in groupby(
                [e.code for e, _, s in forest], 
                key=lambda x: str(jstatutree.lawdata.ReikiCode(x))
            )
        ]
        #print(telems)
        qcodes = []
        code_pairs = [
            (qcodes.append(qcode) or qcode, telement.code, score)
            for telement, _, score in forest
            for qcode in [lca(*telement.attrib["score"].keys())]
        ]
        #qelem = self.find_elem(lca(*qcodes)) if len(qcodes) else self.bind_by_lca(*self.queries.keys())
        qelem = self.bind_by_lca(*qcodes) if len(qcodes) else self.bind_by_lca(*self.queries.keys())
        sub_code_pairs = [edge
                  for telement, edges ,score in forest
                  for edge in edges
                  if edge not in code_pairs
                 ]
        #print(code_pairs)
        return jstatutree.graph.cptable_graph(qelem, telems, code_pairs, sub_code_pairs, tree_name_factory=self._tree_name_factory, layout=layout, **kwargs)

    def calc_element_score(self, element, prox_method='max', *, return_proximity=False, return_diff=False, ):
        score_sum = 0
        rtags = []
        qtags = []
        scoring_dict ={}
        edges = []
        for qtag, (rtag, score) in element.attrib['score'].items():
            scoring_dict[rtag] = max(scoring_dict.get(rtag, (None, -999)), (qtag, score), key=lambda x: x[1])
        for rtag, (qtag, score) in scoring_dict.items():
            rtags.append(rtag)
            qtags.append(qtag)
            score_sum += score
            edges.append((qtag, rtag, round(score, 3)))
        proximity_score = self.calc_element_proximity_score(element, edges, method=prox_method)
        qlca = lca(*qtags)
        tree_sufficiency_score = len(qtags)/sum(1 for qtag in self.find_elem(qlca).iterXsentence(include_code=True, include_value=False))
        score_mean = score_sum/len(rtags)
        #score = round((score_mean + tree_sufficiency_score)/2 * proximity_score, 3)
        score = round((tree_sufficiency_score + proximity_score)/2, 3)
        #print(element.code, score, round(score_sum, 3), round(proximity_score, 3), round(tree_sufficiency_score, 3))
        ret = [edges, score]
        if return_proximity:
            ret.append(proximity)
        if return_diff:
            diff = {
                'plus': list(c for c in element.iterXsentence(include_code=True, include_value=False) if c not in rtags),
                'minus': list(c for c in self.find_elem(qlca).iterXsentence(include_code=True, include_value=False) if c not in qtags)
            }
            ret.append(diff)
        return tuple(ret)
    
    def calc_diff_ranking(self, *, topn=5, prox_method='max', return_diff=False):
        ranking = {}
        for tree in self.match_trees.values():
            ranking.update(self.calc_diff_elements(tree.getroot(), prox_method=prox_method))
        return sorted([(key, score) for key, (elem, score) in ranking.items()], key=lambda x: -x[1])[:topn]
    
    def calc_diff_elements(self, element, *, prox_method='max'):
        if 'score' not in element.attrib:
            #print("skip:", element.code)
            element.attrib['matching_score'] = 0
            return {}
        edge, score = self.calc_element_score(element, prox_method=prox_method)
        element.attrib['matching_score'] = score
        #print("score:", element.code, score)
        if len(element.attrib['score']) <= 1 or len(set(rtag for (rtag, sim) in element.attrib['score'].values())) <= 1:
            return {}
        diff_elements = {}
        new_diff_elements = []
        for c in element:
            diff_elements.update(self.calc_diff_elements(c, prox_method=prox_method))
            if score >= c.attrib['matching_score']:
                #print("hit:", c.code)
                new_diff_elements.append(c)
        csum_score = math.log(len(list(element.iterXsentence(include_code=True, include_value=False))))
        for nde in new_diff_elements: 
            fall_score = score*(sum(c.attrib['matching_score'] for c in element)/len(list(element)) - nde.attrib['matching_score'])
            #print(fall_score, score, nde.attrib['matching_score'])
            diff_elements.update({nde.code: (nde, csum_score*fall_score)})
        return diff_elements
    
    def get_match_elements(self, element, threshold, *, prox_method='max', return_diff=False):
        if 'score' not in element.attrib or len(element.attrib['edges']) <= 1:
            return []
        item = self.calc_element_score(element, prox_method=prox_method, return_diff=return_diff)
        if item[1] > threshold:
            return [(element, *item)]
        else:
            return [e for c in list(element) for e in self.get_match_elements(c, threshold, prox_method=prox_method, return_diff=return_diff)]

    def calc_tree_sufficiency_score(self, qlca):
        return 1/sum(1 for qtag in self.find_elem(qlca).iterXsentence(include_code=True, include_value=False))
        
    def calc_one_way_proximities(self, distances, probabilities):
        prox = [distances[0]]
        for prob, d in zip(probabilities, distances[1:]):
            prox.append((prox[-1]+1)*(1-prob) + d)
        return np.array(prox)
    
    def calc_bidirectional_proximities(self, distances, probabilities):
        return self.calc_one_way_proximities(distances, probabilities) \
                + self.calc_one_way_proximities(distances[::-1], probabilities[::-1])[::-1] \
                - np.array(distances)
    
    def calc_element_proximity_score(self, root, edges, method='max'):
        xsentences = list(root.iterXsentence(include_code=True, include_value=False))
        depth_list = list(len(Path(code).parts) for code in xsentences)
        highest_depth = min(depth_list)
        decay = 0.5
        distances = [0]
        probabilities = []
        for code, depth in zip(xsentences, depth_list):
            for qtag, rtag, score in edges:
                if code == rtag:
                    distances.append(0)
                    probabilities.append(score)
                    break
            else:
                distances[-1] += decay**(depth - highest_depth)
        if "bool" not in method:
            proximities = self.calc_bidirectional_proximities(distances, probabilities)
        else:
            proximities = np.array(distances)
        #print(root.code)
        #print(proximities)
        #print(proximities-np.array(distances))
        #print(probabilities)
        if "smax" in method:
            return 1/(1+max(proximities))
        if 'max' in method:
            #return (len(xsentences) - max(proximities))/len(xsentences)
            return 1/math.sqrt(max(proximities) + 1)
        if 'mean' in method:
            return 1/math.sqrt(sum(proximities)/len(proximities) + 1)

class BUPolicyMatchObject(PolicyMatchObject):
        
    def get_submatch(self, telement, minimum_query=True):
        if isinstance(telement, str):
            telement = self.find_elem(telement)
        elif isinstance(telement, jstatutree.Jstatutree):
            telement = telement.getroot()
        elif isinstance(telement, jstatutree.element.Element):
            telement = telement
        else:
            raise ValueError("Invalid Argument: {}".format(telement))
        sub_match_obj = BUPolicyMatchObject()
        if minimum_query:
            if telement.attrib.get("score", -1) > 0:
                qcode = telement.attrib["qlca"]
                sub_match_obj.queries = {qcode:self.find_elem(qcode)}
            else:
                sub_match_obj.queries = {}
        else:
            sub_match_obj.queries = self.queries
        sub_match_obj.query_tree = self.query_tree
        key = str(jstatutree.lawdata.ReikiCode(telement.code))
        match_tree = copy(self.match_trees[key])
        match_tree._root = telement
        sub_match_obj.match_trees = {key: match_tree}
        return sub_match_obj
    
    def get_match_elements(self, element, threshold, *, prox_method='max', return_diff=False):
        if 'score' not in element.attrib or len(element.attrib.get('edges', [])) <= 1:
            return []
        item = [element.attrib['edges'], round(element.attrib['score'], 3)]
        if item[1] > threshold:
            return [(element, *item)]
        else:
            return [e for c in list(element) for e in self.get_match_elements(c, threshold, prox_method=prox_method, return_diff=return_diff)]
        
    def comp_table(self, threshold, prox_method="max", layout='dot', topn=5, **kwargs):
        forest = self.calc_policy_forest(threshold, prox_method=prox_method, topn=topn)
        if len(forest) == 0:
            print('Warning: comptable is empty')
            return jstatutree.graph.cptable_graph(
                self.query_tree.getroot(), 
                [tree.getroot() for tree in self.match_trees.values()], 
                [], [], tree_name_factory=self._tree_name_factory, layout=layout, **kwargs)
        telems = [
            self.bind_by_lca(*sub_keys)
            for rc, sub_keys in groupby(
                [e.code for e, _, s in forest], 
                key=lambda x: str(jstatutree.lawdata.ReikiCode(x))
            )
        ]
        #print(telems)
        qcodes = []
        code_pairs = [
            (qcodes.append(telement.attrib["qlca"]) or telement.attrib["qlca"], telement.code, score)
            for telement, _, score in forest
        ]
        #qelem = self.find_elem(lca(*qcodes)) if len(qcodes) else self.bind_by_lca(*self.queries.keys())
        qelem = self.bind_by_lca(*qcodes) if len(qcodes) else self.bind_by_lca(*self.queries.keys())
        sub_code_pairs = [edge
                  for telement, edges ,score in forest
                  for edge in telement.attrib['edges']
                  if edge not in code_pairs
                 ]
        #print(code_pairs)
        return jstatutree.graph.cptable_graph(qelem, telems, code_pairs, sub_code_pairs, tree_name_factory=self._tree_name_factory, layout=layout, **kwargs)
        
        

def step(x, th): return 1/(1+math.exp((th-x)*10000)),
def relu(x, th): return x if x>=th else 0.0
def sigmoid_1g(x, th): return 1/(1+math.exp((th-x)*1))
def sigmoid_3g(x, th): return 1/(1+math.exp((th-x)*3))
def sigmoid_5g(x, th): return 1/(1+math.exp((th-x)*5))
def sigmoid_10g(x, th): return 1/(1+math.exp((th-x)*10))
def sigmoid_100g(x, th): return 1/(1+math.exp((th-x)*100)),
        
ACTIVATION_FUNCS = {
    'step': step,
    'ReLU': relu,
    'sigmoid': sigmoid_10g,
    'sigmoid-g1': sigmoid_1g,
    'sigmoid-g3': sigmoid_3g,
    'sigmoid-g5': sigmoid_5g,
    'sigmoid-g10': sigmoid_10g,
    'sigmoid-g100': sigmoid_100g,
}      
class PolicyMatchFactory(object):
    def __init__(self, queries, theta, activation_func=None):
        self.queries = {e.code:e if isinstance(e, jstatutree.element.Element) else e.getroot() for e in queries}
        self.theta = theta
        self.set_activation_func(activation_func)
        self.rev_match_leaves = {}
        self.leaf_keys = []
    
    def set_activation_func(self, func='step'):
        if type(func) is type(lambda x: x):
            self.activation_func = func
        if isinstance(func, str):
            self.activation_func = ACTIVATION_FUNCS[func]

    def calc_elem_score(self, similarity, cip):
        return self.activation_func(similarity, self.theta) * cip

    def add_leaf(self, qtag, rtag, similarity, cip):
        self.rev_match_leaves[rtag] = self.rev_match_leaves.get(rtag, [])
        self.rev_match_leaves[rtag] += [(qtag, self.calc_elem_score(similarity, cip))]#[(qtag, similarity)]
        qtag not in self.leaf_keys and self.leaf_keys.append(qtag)
        return 
    
    @classmethod        
    def aggregate_elem_scores(self, elem, rev_match_leaves):
        ret_scores = {}
        for rtag, scores in rev_match_leaves.items():
            if elem.code not in rtag:
                continue
            for qtag, score in scores:
                cur_score = ret_scores.get(qtag, [None, -1])
                ret_scores[qtag] = (rtag, score) if score > cur_score[1] else cur_score
        return ret_scores
    
    @classmethod
    def scoring_match_tree(cls, arg, rev_match_leaves):
        if isinstance(arg, jstatutree.Jstatutree):
            #print("receive Jstatutree", arg.lawdata.code, list(arg.))
            elem = arg.getroot()
        elif isinstance(arg, jstatutree.element.Element):
            #print("receive Element", arg.code)
            elem = arg
        else:
            raise ValueError("Invalid argument "+str(arg))
        score = cls.aggregate_elem_scores(elem, rev_match_leaves)
        rtags = set(rtag for (rtag, sim) in score.values())
        #if len(score) <= 1 or len(rtags) <= 1:
        #    return
        if len(score) < 1:
            return
        elem.attrib["score"] = score
        for c in list(elem):
            for rtag in rtags:
                if c.code in rtag:
                    cls.scoring_match_tree(c, rev_match_leaves)
                    break
        return arg
            
    def construct_matching_object(self, tree_factory, object_factory=PolicyMatchObject):
        print('Construct Matching Object')
        match_obj = object_factory()
        match_obj.queries = self.queries
        match_obj.query_tree = tree_factory(jstatutree.lawdata.ReikiCode(list(self.queries.keys())[0]))
        """
        for key in self.leaf_keys:
            match_obj.match_leaves[key] = sorted(
                [(rtag, score) for rtag, items in self.rev_match_leaves.items() for qtag, score in items if qtag == key], 
                key=lambda x: -x[1]
            )
        """
        for rc in set(str(jstatutree.lawdata.ReikiCode(rtag)) for rtag in self.rev_match_leaves.keys()):
            tree = tree_factory(rc)
            self.scoring_match_tree(tree, self.rev_match_leaves)
            if len(tree.getroot().get('score', {})) > 1:
                match_obj.match_trees[rc] = tree
                #print('Add candidate tree:', rc)
        return match_obj
            
import networkx as nx
from networkx.algorithms import bipartite

class BUPolicyMatchFactory(object):
    def __init__(self, queries, theta, activation_func=None, decay=0.5):
        self.decay = decay
        self.queries = {e.code:e if isinstance(e, jstatutree.element.Element) else e.getroot() for e in queries}
        self.theta = theta
        self.set_activation_func(activation_func)
        self.rev_match_leaves = {}
        self.leaf_keys = []
        self.qelem_weight_sum_cache = {}
        self.depths = {ec:len(Path(ec).parts) for r in self.queries.values() for ec in r.iterXsentence_code()}

    def set_activation_func(self, func='step'):
        if type(func) is type(lambda x: x):
            self.activation_func = func
        if isinstance(func, str):
            self.activation_func = ACTIVATION_FUNCS[func]

    def calc_elem_score(self, similarity, cip):
        return self.activation_func(similarity, self.theta) * cip

    def add_leaf(self, qtag, rtag, similarity, cip):
        #print(qtag, rtag, similarity)
        self.rev_match_leaves[rtag] = self.rev_match_leaves.get(rtag, [])
        self.rev_match_leaves[rtag] += [(qtag, self.calc_elem_score(similarity, cip))]#[(qtag, similarity)]
        qtag not in self.leaf_keys and self.leaf_keys.append(qtag)
        return 
    
    def find_in_queries(self, code):
        for k , elem in self.queries.items():
            if not k in code:
                print(k, code)
                continue
            return elem.find_by_code(code)
        raise ValueError(str(code))
    
    def matching_node(self, elem):
        qts, rts = [], []
        new_edges = []
        G=nx.Graph()
        for qt, rt, s in edges:
            G.add_node(qt, bipartite=0)
            G.add_node(rt, bipartite=1)
            G.add_edge(qt,rt,weight=s)
        for rt, qt in nx.max_weight_matching(G):
            if rt not in self.rev_match_leaves:
                tmp = rt
                rt = qt
                qt = tmp
            for t, s in self.rev_match_leaves[rt]:
                if qt == t:
                    new_edges.append((qt, rt, s))
                    qts.append(qt)
                    rts.append(rt)
                    break
        return new_edges, rts, qts
    
    def scoring_core(self, elem, edges, rematch=True):
        qts, rts = [], []
        if rematch:
            edges, qts, rts = matching_node(self, elem)
            if len(elem.attrib['edges']) == 0:
                elem.attrib['score'] = 0
                return 
        else:
            elem.attrib['edges'] = edges
            for qt, rt, s in edges:
                qts.append(qt)
                rts.append(rt)
        elem.attrib['qlca'] = lca(*[qt for qt, rt, s in elem.attrib['edges']])
        if elem.attrib['qlca'] not in self.depths:
            self.depths[elem.attrib['qlca']] =  len(Path(elem.attrib['qlca']).parts)
        self.depths[elem.code] = len(Path(elem.code).parts)
        numerator = 0.0
        for qt, rt, s in elem.attrib['edges']:
            telem = elem.find_by_code(rt)
            if telem is None:
                print(rt, qt)
                raise ValueError(rt)
            qelem = self.find_in_queries(qt)
            if qelem is None:
                print(rt, qt)
                raise ValueError(qt)
            numerator += math.sqrt(qelem.attrib.get('weight', 1.0)*(self.decay**(self.depths[rt]-self.depths[elem.code]))*telem.attrib.get('weight', 1.0)*(self.decay**(self.depths[qt]-self.depths[elem.attrib["qlca"]])) )
        qdenominator = sum(
            (qe.attrib.get('weight', 1.0)*self.decay**(self.depths[xskey]-self.depths[elem.attrib['qlca']])) 
            for xskey, qe in self.find_in_queries(elem.attrib['qlca']).iterXsentence_elem(include_code=True) if xskey not in qts
        )
        tdenominator = sum(
            (e.attrib.get('weight', 1.0) * self.decay**(self.depths.get(k, len(Path(k).parts))-self.depths[elem.code])) 
            for k, e in elem.iterXsentence_elem(include_code=True) if k not in rts
        )
        elem.attrib['score'] = numerator/(math.sqrt((1+qdenominator)*(1+tdenominator))-1+numerator)

    
        
    def scoring_match_tree(self, arg, rev_match_leaves):
        if isinstance(arg, jstatutree.Jstatutree):
            #print("receive Jstatutree", arg.lawdata.code, list(arg.))
            elem = arg.getroot()
        elif isinstance(arg, jstatutree.element.Element):
            #print("receive Element", arg.code)
            elem = arg
        else:
            raise ValueError("Invalid argument "+str(arg))
        match_leaf = max(rev_match_leaves.get(elem.code, [(None, -1)]), key=lambda x: x[0])
        if match_leaf[1] >= 0:
            elem.attrib['edges'] = [(qtag, elem.code, sim) for qtag, sim in [match_leaf]]
            elem.attrib['score'] = elem.attrib.get('weight', 1)
            elem.attrib['qlca'] = match_leaf[0]
            self.depths[elem.code] = len(Path(elem.code).parts) 
            return
        for c in elem:
            self.scoring_match_tree(c, self.rev_match_leaves)
        edges = [e for c in elem for e in c.attrib.get('edges', [])]
        if len(edges) ==0:
            elem.attrib['edges'] = []
            return
        qtags = [e[0] for e in edges]
        rtags = [e[1] for e in edges]
        
        rematch = (len(qtags)+len(rtags) > len(set(qtags))+len(set(rtags)))
        self.scoring_core(elem, edges, rematch)
            
    def construct_matching_object(self, tree_factory, object_factory=BUPolicyMatchObject):
        print('Construct Matching Object')
        match_obj = BUPolicyMatchObject()
        match_obj.queries = self.queries
        match_obj.query_tree = tree_factory(jstatutree.lawdata.ReikiCode(list(self.queries.keys())[0]))
        """
        for key in self.leaf_keys:
            match_obj.match_leaves[key] = sorted(
                [(rtag, score) for rtag, items in self.rev_match_leaves.items() for qtag, score in items if qtag == key], 
                key=lambda x: -x[1]
            )
        """
        counted_rc = Counter(str(jstatutree.lawdata.ReikiCode(rtag)) for rtag in self.rev_match_leaves.keys())
        for rc, count in counted_rc.items():
            if count <= 1:
                continue
            tree = tree_factory(rc)
            self.scoring_match_tree(tree, self.rev_match_leaves)
            if len(tree.getroot().attrib['edges']) > 1:
                match_obj.match_trees[rc] = tree
                #print('Add candidate tree:', rc)
        return match_obj
