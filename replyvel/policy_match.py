import re
import math
from pathlib import Path
from itertools import groupby
import numpy as np
import jstatutree



def lca(*keys):
    #print(*keys)
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
    
    def calc_policy_forest(self, threshold, *, topn=5, prox_method='max'):
        forest = []
        for tree in self.match_trees.values():
            forest.extend(self.get_match_elements(tree.getroot(), threshold, prox_method))
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
    
    def split_into_submatches(self, threshold, *, prox_method="max", topn=10):
        for telement, _, score in self.calc_policy_forest(threshold, prox_method=prox_method, topn=topn):
            qcode = lca(* telement.attrib["score"].keys())
            sub_match_obj = PolicyMatchObject()
            sub_match_obj.queries = {qcode:self.find_elem(qcode)}
            sub_match_obj.query_tree = self.query_tree
            key = str(jstatutree.lawdata.ReikiCode(telement.code))
            sub_match_obj.match_trees = {key: self.match_trees[key]}
            yield sub_match_obj
    
    def comp_table(self, threshold, prox_method="max", layout='dot', topn=5, **kwargs):
        forest = self.calc_policy_forest(threshold, prox_method=prox_method, topn=topn)
        
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
        qelem = self.find_elem(lca(*qcodes)) if len(qcodes) else self.bind_by_lca(*self.queries.keys())
        sub_code_pairs = [edge
                  for telement, edges ,score in forest
                  for edge in edges
                  if edge not in code_pairs
                 ]
        #print(code_pairs)
        return jstatutree.graph.cptable_graph(qelem, telems, code_pairs, sub_code_pairs, tree_name_factory=self._tree_name_factory, layout=layout, **kwargs)

    def calc_element_score(self, element, prox_method='max', return_proximity=False):
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
        proximity_score = self.calc_element_proximity_score(element, edges, prox_method)
        tree_sufficiency_score = self.calc_tree_sufficiency_score(qtags)
        score = round(score_sum * proximity_score * tree_sufficiency_score, 3)
        #print(element.code, score, round(score_sum, 3), round(proximity_score, 3), round(tree_sufficiency_score, 3))
        return (score, edges) if not return_proximity else (score, edges, proximity)
    
    def get_match_elements(self, element, threshold, prox_method='max'):
        if 'score' not in element.attrib or len(element.attrib['score']) <= 1 or len(set(rtag for (rtag, sim) in element.attrib['score'].values())) <= 1:
            return []
        score, edges = self.calc_element_score(element, prox_method)
        if score > threshold:
            return [(element, edges, score)]
        else:
            return [e for c in list(element) for e in self.get_match_elements(c, threshold)]

    def calc_tree_sufficiency_score(self, qtags):
        qlca = lca(*qtags)
        return 1/sum(1 for qtag in self.find_elem(qlca).iterXsentence(include_code=True, include_value=False))
        
    def calc_element_proximity_score(self, root, edges, method='max'):
        proximities = [0.0]
        #print(leaf_tags)
        #print(list(root.iterXsentence(include_code=True, include_value=False)))
        xsentences = list(root.iterXsentence(include_code=True, include_value=False))
        for code in xsentences:
            for qtag, rtag, score in edges:
                if code == rtag:
                    proximity = (proximities[-1]+1) * (1.0-score)
                    break
            else:
                proximity = proximities[-1] + 1
            proximities.append(proximity)
        proximities_arr = np.array(proximities[1:])
        proximities = [0.0]
        for code in xsentences[::-1]:
            for qtag, rtag, score in edges:
                if code == rtag:
                    proximity = (proximities[-1]+1) * (1.0-score)
                    break
            else:
                proximity = proximities[-1] + 1
            proximities.append(proximity)
        proximities_arr += np.array(proximities[1:][::-1])
        proximities_arr /=2
        proximities = proximities_arr
        #print(root.code, proximities)
        print(proximities)
        if method is 'max':
            return (len(proximities) - max(proximities))/len(proximities)
        if method is 'mean':
            return (( len(proximities) ** 2)/2- sum(proximities))/len(proximities)*2
        
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
            if func == 'step':
                self.activation_func = lambda x: 1.0 if x>=0 else 0.0
            elif func == 'ReLU':
                self.activation_func = lambda x: x if x>=0 else 0.0
            elif func == 'sigmoid-g1':
                self.activation_func = lambda x: 1/(1+math.exp(-x*1))
            elif func in ('sigmoid-g10', 'sigmoid'):
                self.activation_func = lambda x: 1/(1+math.exp(-x*10))
            elif func == 'sigmoid-g100':
                self.activation_func = lambda x: 1/(1+math.exp(-x*100))

    def calc_elem_score(self, similarity, cip):
        return self.activation_func(similarity-self.theta) * cip

    def add_leaf(self, qtag, rtag, similarity, cip):
        self.rev_match_leaves[rtag] = self.rev_match_leaves.get(rtag, [])
        self.rev_match_leaves[rtag] += [(qtag, self.calc_elem_score(similarity, cip))]#[(qtag, similarity)]
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
            self.scoring_match_tree(tree.getroot())
            if len(tree.getroot().get('score', {})) > 1:
                match_obj.match_trees[rc] = tree
                print('Add candidate tree:', rc)
        return match_obj


