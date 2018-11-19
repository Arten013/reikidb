import re
import math
from jstatutree.lawdata import ReikiCode

class PolicyMatchObject(object):
    def __init__(self, db, threshold, activation_func=None):
        self.db = db
        self.match_trees = {}
        self.match_leaves = {}
        #self.decay = decay
        self.threshold = threshold
        self.set_activation_func(activation_func)

    def set_activation_func(self, func='ReLU'):
        if type(func) is type(lambda x: x):
            self.activation_func = func
        if isinstance(func, str):
            if func == 'ReLU':
                self.activation_func = lambda x: 1.0 if x>=0 else 0.0
            elif func == 'sigmoid':
                self.activation_func = lambda x: 1/(1+math.exp(-x))

    def calc_elem_score(self, similarity, cip):
        return self.activation_func(similarity-self.threshold) * cip

    def add_leaf(self, qtag, rtag, similarity, cip):
        self.match_leaves[qtag] = self.match_leaves.get(qtag, [])
        self.match_leaves[qtag].append((rtag, similarity))
        rc = str(ReikiCode(rtag))
        if rc not in self.match_tree:
            self.match_tree = self.db.get_jstatutree(rc)
        elem = self.match_tree[rc].getroot()
        score = self.calc_elem_score(similarity, cip)
        while len(list(elem)):
            if 'score' not in elem.attr:
                elem.attr['score'] = {qtag: (rtag, score)}
            else:
                if elem.attr['score']['qtag'][1] < score:
                    elem.attr['score']['qtag'] = (rtag, score)
            for c in list(elem):
                if c.code in rtag:
                    elem = c
                    break
            else:
                raise Exception('Unexpected Error')
        return leaf

    def calc_policy_forest(self, threshold):
        forest = []
        for tree in self.match_trees.values():
            forest.extend(self.get_match_elements(tree.getroot(), threshold, prox_method))
        return forest

    def get_match_elements(self, element, threshold, prox_method):
        if 'score' not in element.attr::
            return []
        score_sum = 0
        rtags = 0
        for k, (rtag, score) in element.attr['score'].items():
            score_sum += score
            rtags.append(rtag)
        score = score_sum * self.calc_element_proximity(element, rtags, prox_method)
        if score > threshold:
            return [element]
        else:
            return [e for c in list(element) for e in self.get_match_elements(c)]

    def calc_element_proximity(self, root, leaf_tags, method='max'):
        proximities = []
        proximity = 0
        for code ,s in root.iterXsentence(include_code=True):
            if code not in leaf_tags:
                proximity += 1
                continue
            proximities.append(proximity)
        proximities.append(proximity)
        if method is 'max':
            return max(proximities)
        if method is 'mean':
            return sum(proximities)/len(proximities)
        if method is 'sum':
            return sum(proximities)

            



