import os
import re
import shutil
import concurrent
from itertools import groupby
from pathlib import Path
from xml.etree import ElementTree as ET, ElementPath
from gensim.models.doc2vec import TaggedDocument

from jstatutree import Jstatutree
from jstatutree.exceptions import LawError
from jstatutree.element import Element
from jstatutree.etypes import ETYPES, code2etype, code2jname, CATEGORY_TEXT
from jstatutree.lawdata import ReikiCode
from multiprocessing import cpu_count

from . import _replyvel as replyvel

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline
from sklearn.neighbors import NearestNeighbors
from sklearn.externals import joblib
import numpy as np
import pickle


class TaggedVectors(object):
    def __init__(self, vector_size):
        self.vector_size = vector_size
        self.tagged_indice = {}
        self.indexed_tags = []
        self.vectors = None
    
    def _add_tags(self, *tags):
        for tag in tags:
            self.tagged_indice[tag] = len(self.indexed_tags)
            self.indexed_tags.append(tag)
    
    def add(self, vec, tag):
        self.add_tag(tag)
        self.vectors = np.vstack([self.vectors, vec]) if self.vectors is not None else vec
    
    def add_from_tags_and_vectors(self, tags, vectors):
        self.vectors = np.vstack([self.vectors, vectors]) if self.vectors is not None else vectors
        self._add_tags(*tags)
    
    def add_from_keyed_vectors(self, iterator):
        additional_vectors = np.array([vec for tag, vec in iterator if (self._add_tags(tag) or True)])
        self.vectors = np.vstack([self.vectors, additional_vectors]) if self.vectors is not None else additional_vectors
    
    def __getitem__(self, item):
        if isinstance(item, str):
            return self.vectors[x]
        elif isinstance(item, int):
            return self.vectors[self.tagged_indice[x]]
        raise KeyError('Invalid type {} for tagged vector key.'.format(item.__class__))
    
    def knn_by_keys(self, keys):
        return self.knn(self.vectors[(self.tagged_indice[k] for key in keys)])
    
    def __len__(self):
        return self.vectors.shape[0] if self.vectors is not None else 0
    
    def knn(self, query_vectors, k=10):
        distances_list, indice_list = NearestNeighbors(
            n_neighbors=k, 
            metric='cosine', 
            algorithm='brute'
        ).fit(self.vectors).kneighbors(query_vectors)
        return [list(zip([self.indexed_tags[i] for i in indice], [round(d, 3) for d in distances])) 
                        for distances, indice in zip(distances_list, indice_list)]

def get_ngram_set(s, n):
    s = '▁'*(n-1) + s + '▁'*(n-1)
    return set([s[i:i+n] for i in range(len(s)-(n-1))])
import math
class SimStringModule(object):
    MODEL_TYPE_TAG = ''
    OLAP_FUNCTIONS = {
        'cosine': lambda x, y, n:  round([ len(nx&ny) / math.sqrt( len(nx)*len(ny) )  for nx in [get_ngram_set(x, n)] for ny in [get_ngram_set(y, n)] ][0], 3)
    }
    def __init__(self, *args, **kwargs):
        super().__init__(*args, *kwargs)
        self._simstring = None
        self.simstring_path = None
        self.simstring_measure = None
    
    def set_simstring_property(self, path=None, measure='cosine'):
        self.simstring_path = path or self.simstring_path
        self.simstring_measure = measure
    
    @property
    def simstring(self):
        if self._simstring is None:
            self.simstring =  simstring.reader(self.simstring_path)
            self.simstring.measure = getattr(simstring, self.simstring_measure )
        return self._simstring
        
    def save(self):
        if self._simstring is not None:
            self._simstring = None
            super().save()
            self.simsting
        else:
            super().save()

    def most_similar_by_keys(self, keys, topn=10, name_as_tag=True, append_text=False, append_olap=True, olap_measure='cosine', ngram=3):
        ukeys, query_vectors = self.keys_to_vectors(*keys, return_keys=True)
        olap_func = self.OLAP_FUNCTIONS.get(olap_measure, 'cosine')
        if not len(self.tagged_vectors):
            self.reload_tagged_vectors()
        ret = []
        for qtag, tag_dist_pairs in zip(ukeys, self.tagged_vectors.knn(query_vectors, k=topn)):
            #print(tag_dist_pairs)
            ranking = tag_dist_pairs
            tag2sent = lambda x: ''.join([re.sub('▁', '', w) for s in self.db.get_element(x).itersentence() for w in s])
            if append_text:
                ranking = (
                    (t, d, tag2sent(t), olap_func(tag2sent(t), tag2sent(qtag), ngram)) if append_olap else (t, d, tag2sent(t))
                      for t, d in ranking  
                )
            elif append_olap:
                ranking = (
                    (t, d, olap_func(tag2sent(t), tag2sent(qtag), ngram))
                      for t, d in ranking  
                )
            if name_as_tag:
                ranking = ((self.db.get_element_name(x[0], x[0]), *x[1:]) for x in ranking)
            ret.append(list(ranking))
        return zip(ukeys, ret)
    
class JstatutreeModel(object):
    MODEL_TYPE_TAG = ''
    def __init__(self, tdb, tag, vector_size, unit='XSentence'):
        assert hasattr(tdb, 'tokenizer'), 'TokenizedJstatutreeDB required'
        assert len(self.MODEL_TYPE_TAG), 'You must set cls.MODEL_TYPE_TAG in the model implementation.'
        self.db = tdb
        self.tag = tag
        self.unit = unit
        self.vector_size = vector_size
        os.makedirs(self.path, exist_ok=True)
        self.vecs = replyvel.DB(Path(self.path, 'db'), target_codes=self.db.target_codes)
        self.tagged_vectors = TaggedVectors(vector_size)
        self.rspace_codes = list(tdb.mcodes)
    
    def save(self):
        with open(Path(self.path, 'jsmodel.pkl'), 'wb') as f:
            joblib.dump(self, f)
            
    @classmethod
    def load(cls, path):
        with open(Path(path, 'jsmodel.pkl'), 'rb') as f:
            return joblib.load(f)
    
    def fit(self):
        raise 'Implementation error'
    
    def _calc_vec(self, text):
        raise 'Implementation error'
        
    @property
    def path(self):
        return Path(self.db.path, self.MODEL_TYPE_TAG, self.tag)
    
    def restrict_rspace(self, rspace_codes):
        self.rspace_codes = rspace_codes
        self.reload_tagged_vectors(rspace_codes)
    
    def reload_tagged_vectors(self, rspace_codes=None):
        self.tagged_vectors = TaggedVectors(self.vector_size)
        self.tagged_vectors.add_from_keyed_vectors(self.vecs.get_subdb(target_codes=rspace_codes or self.rspace_codes or self.vecs.mcodes).iterator())
    
    def get_submodel(self, *mcodes):
        submodel = self.__class__(self.db.get_subdb(mcodes), self.tag, self.vector_size, self.unit)
        submodel.reload_tagged_vectors()
        return submodel
    
    def remove_files(self):
        if self.path.exists():
            print('remove dir:', self.path)
            shutil.rmtree(self.path)
    
    def most_similar(self, query_vectors, topn=10, name_as_tag=True, append_text=False):
        if not len(self.tagged_vectors):
            self.reload_tagged_vectors()
        ret = []
        for tag_dist_pairs in self.tagged_vectors.knn(query_vectors, k=topn):
            #print(tag_dist_pairs)
            ranking = tag_dist_pairs
            if append_text:
                ranking = (
                    (t, d, ''.join([w for s in self.db.get_element(t).itersentence() for w in s]))
                    for t, d in ranking)
            if name_as_tag:
                ranking = ((self.db.get_element_name(x[0], x[0]), *x[1:]) for x in ranking)
            ret.append(list(ranking))
        return ret
    
    def keys_to_vectors(self, *keys, **kwargs):
        return_keys = kwargs.get('return_keys', False)
        ukeys = []
        for key in keys:
            try:
                tree = self.db.get_jstatutree(key, None)
                if tree:
                    elem = tree._root
                assert tree
            except:
                elem = self.db.get_element(key, None)
                assert elem, 'invalid key: '+str(key)
            if self.unit == 'XSentence':
                ukeys.extend([c for c, _ in elem.iterXsentence(include_code=True)])
            else:
                ukeys.extend([c for c, _ in elem.iterfind('.//{unit}'.format(unit=self.unit))])
        vectors = np.array([self.vecs.get(k) for k in ukeys])
        print(vectors.shape)
        return (ukeys, vectors) if return_keys else vectors
        
    def most_similar_by_keys(self, keys, *args, **kwargs):
        ukeys, qvecs = self.keys_to_vectors(*keys, return_keys=True)
        ranking = self.most_similar(qvecs, *args, **kwargs)
        return list(zip(ukeys, ranking))
    
    def get(self, key, default=None):
        ret = self.vecs.get(key, None)
        if ret is not None:
            return ret
        e = self.db.get_element(key, None)
        if e is None:
            return default
        words = [w for s in e.itersentences() for w in s]
        return self._calc_vec(text)
