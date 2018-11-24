import re
import math
from pathlib import Path
from itertools import groupby

import jstatutree



def lca(*keys):
    #print(keys)
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
    
    def calc_policy_forest(self, threshold, prox_method='max'):
        forest = []
        for tree in self.match_trees.values():
            forest.extend(self.get_match_elements(tree.getroot(), threshold, prox_method))
        return forest
    
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
    
    def split_into_submatches(self, threshold, prox_method="max"):
        forest = self.calc_policy_forest(threshold, prox_method)
        for telement, score in sorted(forest, key=lambda x: -x[1]):
            qcode = lca(* telement.attrib["score"].keys())
            sub_match_obj = PolicyMatchObject()
            sub_match_obj.queries = {qcode:self.find_elem(qcode)}
            sub_match_obj.query_tree = self.query_tree
            key = str(jstatutree.lawdata.ReikiCode(telement.code))
            sub_match_obj.match_trees = {key: self.match_trees[key]}
            yield sub_match_obj
    
    def comp_table(self, threshold, prox_method="max", layout='dot', **kwargs):
        forest = self.calc_policy_forest(threshold, prox_method)
        
        telems = [
            self.bind_by_lca(*sub_keys)
            for rc, sub_keys in groupby(
                [e.code for e, s in forest], 
                key=lambda x: str(jstatutree.lawdata.ReikiCode(x))
            )
        ]
        #print(telems)
        qcodes = []
        code_pairs = [
            (qcodes.append(qcode) or qcode, telement.code, score)
            for telement, score in forest
            for qcode in [lca(*telement.attrib["score"].keys())]
        ]
        qelem = self.bind_by_lca(*qcodes) if len(qcodes) else self.bind_by_lca(*self.queries.keys())
        sub_code_pairs = [(qcode, tscode, score)
                  for telement, score in forest
                  for qcode, (tscode, score) in telement.attrib["score"].items()
                  if (qcode, tscode, score) not in code_pairs
                 ]
        #print(code_pairs)
        return jstatutree.graph.cptable_graph(qelem, telems, code_pairs, sub_code_pairs, tree_name_factory=self._tree_name_factory, layout=layout, **kwargs)

    def calc_element_score(self, element, prox_method='max', return_proximity=False):
        score_sum = 0
        rtags = []
        for k, (rtag, score) in element.attrib['score'].items():
            score_sum += score
            rtags.append(rtag)
        proximity = self.calc_element_proximity(element, rtags, prox_method)
        score = score_sum /(score_sum+ proximity)
        return score if not return_proximity else (score, proximity)
    
    def get_match_elements(self, element, threshold, prox_method='max'):
        if 'score' not in element.attrib or len(element.attrib['score']) <= 1 or len(set(rtag for (rtag, sim) in element.attrib['score'].values())) <= 1:
            return []
        score = self.calc_element_score(element, prox_method)
        if score > threshold:
            return [(element, score)]
        else:
            return [e for c in list(element) for e in self.get_match_elements(c, threshold)]

    def calc_element_proximity(self, root, leaf_tags, method='max'):
        proximities = []
        proximity = 0
        #print(leaf_tags)
        #print(list(root.iterXsentence(include_code=True, include_value=False)))
        for code in root.iterXsentence(include_code=True, include_value=False):
            if code not in leaf_tags:
                proximity += 1
                continue
            proximities.append(proximity)
            proximity = 1
        proximities.append(proximity)
        #print(root.code, proximities)
        if method is 'max':
            return max(proximities)
        if method is 'mean':
            return sum(proximities)/len(proximities)
        if method is 'sum':
            return sum(proximities)
        
class PolicyMatchFactory(object):
    def __init__(self, queries, theta, activation_func=None):
        self.queries = {e.code:e if isinstance(e, jstatutree.element.Element) else e.getroot() for e in queries}
        self.theta = theta
        self.set_activation_func(activation_func)
        self.rev_match_leaves = {}
        self.leaf_keys = []
    
    def set_activation_func(self, func='ReLU'):
        if type(func) is type(lambda x: x):
            self.activation_func = func
        if isinstance(func, str):
            if func == 'ReLU':
                self.activation_func = lambda x: 1.0 if x>=0 else 0.0
            elif func == 'sigmoid':
                self.activation_func = lambda x: 1/(1+math.exp(-x))

    def calc_elem_score(self, similarity, cip):
        return self.activation_func(similarity-self.theta) * cip

    def add_leaf(self, qtag, rtag, similarity, cip):
        self.rev_match_leaves[rtag] = self.rev_match_leaves.get(rtag, [])
        self.rev_match_leaves[rtag] += [(qtag, similarity)]
        qtag not in self.leaf_keys and self.leaf_keys.append(qtag)
        return 
            
    def aggregate_elem_scores(self, elem):
        ret_scores = {}
        for rtag, scores in self.rev_match_leaves.items():
            if elem.code not in rtag:
                continue
            for qtag, score in scores:
                cur_score = ret_scores.get(qtag, [None, -1])
                ret_scores[qtag] = (rtag, score) if score > cur_score[1] else cur_score
        return ret_scores
    
    def scoring_match_tree(self, elem):
        score = self.aggregate_elem_scores(elem)
        rtags = set(rtag for (rtag, sim) in score.values())
        if len(score) <= 1 or len(rtags) <= 1:
            return
        elem.attrib["score"] = score
        for c in list(elem):
            for rtag in rtags:
                if c.code in rtag:
                    self.scoring_match_tree(c)
                    break
            
    def construct_matching_object(self, tree_factory, object_factory=PolicyMatchObject):
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
            self.scoring_match_tree(tree.getroot())
            match_obj.match_trees[rc] = tree
        return match_obj


