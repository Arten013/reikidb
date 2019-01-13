import os
import subprocess
import re
import shutil
from pathlib import Path
from sklearn.neighbors import NearestNeighbors
from sklearn.externals import joblib
import numpy as np
import pickle
import math
from multiprocessing import cpu_count

from .policy_match import BUPolicyMatchFactory as PolicyMatchFactory
from . import _replyvel as replyvel
from .dataset import SentenceGenerator
from scipy import cluster, spatial, sparse
from .utils.logger import get_logger
from time import time

class TaggedVectors(object):
    def __init__(self, vector_size, vector_type='numpy_array'):
        self.vector_type = vector_type
        self.vector_size = vector_size
        self.tagged_indice = {}
        self.indexed_tags = []
        self.vectors = None
        self.clusters = None
    
    def _add_tags(self, *tags):
        for tag in tags:
            self.tagged_indice[tag] = len(self.indexed_tags)
            self.indexed_tags.append(tag)
    
    def vstack(self, vectors):
        if self.vector_type == 'numpy_array':
            return np.vstack(vectors)
        elif self.vector_type == 'scipy_sparse_matrix':
            try:
                return sparse.vstack(vectors)
            except:
                get_logger('TaggedVectors.vstack').critical('Invalid Shape: %s', str(vectors))
        raise Exception('Invalid vector_type '+str(self.vector_type))
    
    def add(self, vec, tag):
        self.add_tag(tag)
        self.vectors = self.vstack([self.vectors, vec]) if self.vectors is not None else vec
        
    """
    def add_from_tags_and_vectors(self, tags, vectors):
        self.vectors = np.vstack([self.vectors, vectors]) if self.vectors is not None else vectors
        self._add_tags(*tags)
    """
    
    def get_vectors(self, keys):
        return self.vectors[[self.tagged_indice[k] for k in keys]]
    
    def add_from_keyed_vectors(self, iterator):
        additional_vectors = [vec for tag, vec in iterator if (self._add_tags(tag) or True)]
        if len(additional_vectors) == 0:
            return 
        self.vectors = self.vstack([self.vectors, self.vstack(additional_vectors)]) if self.vectors is not None else self.vstack(additional_vectors)
    
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
    
    def fit_linkage_matrix(self, method='single', metric='euclidean', optimal_ordering=False):
        assert len(self), "You must first fit text vectors."
        print("acquire distance matrix")
        pdist = spatial.distance.pdist(self.vectors, metric=metric)
        print("acquire linkage_matrix")
        self.linkage_matrix = cluster.hierarchy.linkage(pdist, method=method, metric=metric, optimal_ordering=optimal_ordering)
        print("Finished!")
        
    def fcluster(self, t, *args, **kwargs):
        assert self.linkage_matrix is not None, "You must first fit linkage matrix using TaggedVectors.fit_linkage_matrix"
        print("flatten cluster")
        self.clusters = cluster.hierarchy.fcluster(self.linkage_matrix, t, *args, **kwargs).tolist()
        print("Finished!")
    
    def fit_kmeans(self, n_clusters=8, init='k-means++', n_init=10, max_iter=300, tol=0.0001, precompute_distances='auto', verbose=0, random_state=None, copy_x=True, n_jobs=None, algorithm='auto'):
        assert len(self), "You must first fit text vectors."
        self.kmeans=KMeans(n_clusters, init, n_init, max_iter, tol, precompute_distances, verbose, random_state, copy_x, n_jobs, algorithm)
        model = self.kmeans.fit(self.vectors)
        self.clusters = model.labels_
        return model
    
    def get_cluster(self, key):
        return self.clusters[self.tagged_indice[key]]
    
    def get_array(self, vectors=None):
        vectors = vectors if vectors is not None else self.vectors
        if self.vector_type == 'numpy_array':
            return vectors
        elif self.vector_type == 'scipy_sparse_matrix':
            return vectors.toarray()
        raise Exception('Invalid vector_type '+str(self.vector_type))
    
    def knn(self, query_vectors, k=10):
        distances_list, indice_list = NearestNeighbors(
            n_neighbors=k, 
            metric='cosine', 
            algorithm='brute',
            n_jobs=cpu_count()
        ).fit(self.get_array()).kneighbors(query_vectors)
        return [list(zip([self.indexed_tags[i] for i in indice], [round(1-d, 3) for d in distances])) 
                        for distances, indice in zip(distances_list, indice_list)]

def get_ngram_set(s, n):
    s = '▁'*(n-1) + s + '▁'*(n-1)
    return set([s[i:i+n] for i in range(len(s)-(n-1))])

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
    
class JstatutreeModelCore(object):
    MODEL_TYPE_TAG = ''
    def __init__(self, db, tag, unit='XSentence'):
        assert len(self.MODEL_TYPE_TAG), 'You must set cls.MODEL_TYPE_TAG in the model implementation.'
        self.db = db
        self.tag = tag
        self.unit = unit
        os.makedirs(self.path, exist_ok=True)
        self.rspace_codes = list(db.mcodes)

    def is_fitted(self):
        raise Exception("Not Implemented")
    
    def save(self):
        with open(Path(self.path, 'jsmodel.pkl'), 'wb') as f:
            joblib.dump(self, f)
            
    @classmethod
    def load(cls, path):
        with open(Path(path, 'jsmodel.pkl'), 'rb') as f:
            return joblib.load(f)
        
    @property
    def rspace_db(self):
        return self.db.get_subdb(self.rspace_codes)
    
    def keys_to_ukeys(self, *keys):
        ukeys = []
        for key in keys:
            try:
                elem = self.db.get_element(key, None)
                assert elem
                
            except:
                tree = self.db.get_jstatutree(key, None)
                if tree:
                    elem = tree._root
                assert tree, 'invalid key: '+str(key)
            if self.unit == 'XSentence':
                ukeys.extend([c for c, _ in elem.iterXsentence(include_code=True)])
            else:
                ukeys.extend([c for c, _ in elem.iterfind('.//{unit}'.format(unit=self.unit))])
        return ukeys

    def fit(self):
        raise 'Implementation error'
        
    @property
    def path(self):
        return Path(self.db.path, self.MODEL_TYPE_TAG, self.tag)
    
    def restrict_rspace(self, rspace_codes):
        self.rspace_codes = rspace_codes

    def get_submodel(self, *mcodes):
        submodel = self.__class__(self.db.get_subdb(mcodes), self.tag, self.vector_size, self.unit)
        return submodel

    def remove_files(self):
        if self.path.exists():
            print('remove dir:', self.path)
            shutil.rmtree(self.path)

    def build_match_factory(self, query_key, theta, match_factory_cls, weight_border=0.5, sample_num=500):
        logger = get_logger(self.__class__.__name__+'.build_match_factory')
        query = self.db.get_jstatutree(query_key).change_root(query_key)
        match_factory = match_factory_cls(query, tree_factory=self.rspace_db.get_jstatutree)
        t = time()
        rankings = self.most_similar_by_keys([query_key], topn=sample_num)
        logger.info('leaf retrieve: %s sec/ %d query leaves', str(round(time()-t, 3)), len(rankings))
        for i, (qtag, ranking) in enumerate(rankings):
            sims = []
            for rtag, similarity in ranking:
                if similarity < theta:
                    break
                logger.debug('%s-%s', str(qtag), str(rtag))
                match_factory.add_leaf(qtag, rtag, similarity)
                sims.append(similarity)
            query.find_by_code(qtag).attrib['weight'] = weight_border/(weight_border+sum(sims))
            logger.info('%03d/%d:add %d leaves similar to %s',i ,len(rankings), len(sims), qtag)
            logger.info('Currently, %d trees stored in the builder.', len(match_factory.tree_store))
            if len(sims) > 0:
                logger.info('Similarity summary: mean %.3f, max %.3f, min %.3f.', sum(sims)/max(len(sims), 1), max(sims), min(sims))
        return match_factory

    def policy_matching(self, keys, theta, activation_func, sample_num=500, **kwargs):
        match_factory = PolicyMatchFactory([self.db.get_element(k) for k in keys], theta, activation_func)
        rankings = self.most_similar_by_keys(keys, topn=sample_num, **kwargs)
        for qtag, ranking in rankings:
            for rtag, similarity in ranking:
                if similarity < theta:
                    break
                match_factory.add_leaf(qtag, rtag, similarity, cip=1.0)
        return match_factory.construct_matching_object(tree_factory = self.db.get_jstatutree)

    def most_similar_by_keys(self, keys, *args, **kwargs):
        raise 'Not Implemented.'

class JstatutreeVectorModelBase(JstatutreeModelCore):
    MODEL_TYPE_TAG = ''
    def __init__(self, db, tag, vector_size, unit='XSentence', vector_type='numpy_array'):
        assert hasattr(db, 'tokenizer'), 'TokenizedJstatutreeDB required'
        super().__init__(db, tag, unit)
        self.vector_size = vector_size
        self.vecs = replyvel.DB(Path(self.path, 'db'), target_codes=self.db.target_codes)
        self.vector_type = vector_type
        self.tagged_vectors = self.tagged_vector_factory()
        
    def tagged_vector_factory(self):
        return TaggedVectors(self.vector_size, vector_type=self.vector_type)
        
    def fit_kmeans(self, n_clusters=8, init='k-means++', n_init=10, max_iter=300, tol=0.0001, precompute_distances='auto', verbose=0, random_state=None, copy_x=True, n_jobs=None, algorithm='auto', reload_tagged_vectors=True):
        reload_tagged_vectors and self.reload_tagged_vectors()
        self.kmeans_model = self.tagged_vectors.fit_kmeans(n_clusters, init, n_init, max_iter, tol, precompute_distances, verbose, random_state, copy_x, n_jobs, algorithm)
        
    def _calc_vec(self, text):
        raise 'Implementation error'
    
    def restrict_rspace(self, rspace_codes):
        self.rspace_codes = rspace_codes
        self.reload_tagged_vectors(rspace_codes)
    
    def reload_tagged_vectors(self, rspace_codes=None):
        self.tagged_vectors = self.tagged_vector_factory()
        self.tagged_vectors.add_from_keyed_vectors(self.vecs.get_subdb(target_codes=rspace_codes or self.rspace_codes or self.vecs.mcodes).iterator())

    def most_similar(self, query_vectors, topn=10, name_as_tag=False, append_text=False):
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
        ukeys = self.keys_to_ukeys(*keys)
        if len(ukeys) == 0:
            return ([], []) if return_keys else []
        vectors = self.tagged_vectors.get_array(self.tagged_vectors.vstack([self.vecs.get(k) for k in ukeys]))
        print(vectors.shape)
        return (ukeys, vectors) if return_keys else vectors
        
    def most_similar_by_keys(self, keys, *args, **kwargs):
        ukeys, qvecs = self.keys_to_vectors(*keys, return_keys=True)
        if len(ukeys) == 0:
            return []
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

    @property
    def training_corpus_path(self):
        return self.path/'training_corpus'

    def get_training_corpus(self):
        if not self.training_corpus_path.exists():
            os.makedirs(self.training_corpus_path, exist_ok=True)
            print('Export corpus file to', self.training_corpus_path)
            self.db.export_corpus(path=self.training_corpus_path, unit=self.unit)
        return self.training_corpus_path
    def is_fitted(self):
        iterator = self.vecs.iterator(include_key=False, include_value=False)
        try:
            print(next(iterator))
            ret = True
        except StopIteration:
            print('not fitted')
            ret = False
        finally:
            del iterator
        return ret

from sklearn.cluster import KMeans
class ScikitModelBase(JstatutreeVectorModelBase):
    def _calc_vec(self, text):
        if isinstance(text, str):
            text = self.db.tokenizer(text)
        return self.transformer.transform(text)
    
    def is_fitted(self):
        iterator = self.vecs.iterator(include_key=False, include_value=False)
        try:
            print(next(iterator))
            ret = True
        except StopIteration:
            print('not fitted')
            ret = False
        finally:
            del iterator
        return ret


    def fit(self, task_size=None): #task_size argument is just for compatibility
        if self.is_fitted():
            print('model already exists.')
            return
        print('Training begin')
        with open(self.get_training_corpus()/'corpus.txt') as f:
            vectors = self.transformer.fit_transform(
                (
                    f
                )
            )
        print(vectors.shape, vectors[0].shape)
        #print(vectors)
        print('Training finished')
        print('Add vectors to replyvel.DB')
        with open(self.get_training_corpus()/'tags.txt') as tags:
            with self.vecs.write_batch() as wb:
                for i, tag in enumerate(tags):
                    tag = tag.rstrip()
                    vector = vectors[i]
                    #if vector.shape != (self.vector_size,):
                        #print('skip:', tag, vector.shape)
                        #continue
                    wb.put(tag, vector)
        print('Model fitting complete!')
        self.save()
