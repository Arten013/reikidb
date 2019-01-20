import os
import re
import shutil
import concurrent
import numpy as np
import pickle
import math
import traceback
import gensim

from . import _replyvel as replyvel

from pathlib import Path
from multiprocessing import cpu_count
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline

from .model_core import JstatutreeModelCore, JstatutreeVectorModelBase, ScikitModelBase
from .dataset import SentenceGenerator
from concurrent.futures import ProcessPoolExecutor
import simstring
import hashlib
from .policy_match import BUPolicyMatchFactory as PolicyMatchFactory
from fastText import train_unsupervised
import joblib
from .utils.logger import get_logger
from time import time

class TfIdf(ScikitModelBase):
    MODEL_TYPE_TAG = 'tfidf'
    def __init__(self, db, tag, vocab_size=8000, idf=True, unit='XSentence'):
        super().__init__(db, tag, vocab_size, unit, vector_type='scipy_sparse_matrix')
        self.vocab_size = vocab_size
        self.idf = idf
        count_vectorizer = CountVectorizer(
            input='content',
            #max_df=0.5, 
            #min_df=1, 
            lowercase = False,
            max_features=self.vocab_size
            )
        steps = [('CountVectorize', count_vectorizer)]
        steps.append(("TfidfTransform", TfidfTransformer(use_idf=idf)))
        self.transformer = Pipeline(steps)

    def get_submodel(self, *mcodes):
        submodel = self.__class__(self.db.get_subdb(mcodes), self.tag, self.vocab_size, self.idf, self.unit)
        if len(self.tagged_vectors):
            submodel.reload_tagged_vectors()
        submodel.transformer = self.transformer
        return submodel


class LSI(ScikitModelBase):
    MODEL_TYPE_TAG = 'lsi'
    def __init__(self, db, tag, vocab_size=16000, idf=True, vector_size=300, unit='XSentence'):
        super().__init__(db, tag, vector_size, unit)
        self.vocab_size = vocab_size
        self.idf = idf
        count_vectorizer = CountVectorizer(
            input='content',
            #max_df=0.5, 
            #min_df=1, 
            lowercase = False,
            max_features=self.vocab_size
            )
        steps = [('CountVectorize', count_vectorizer)]
        steps.append(("TfidfTransform", TfidfTransformer(use_idf=idf)))
        steps.append((
            "TruncatedSVD",
            TruncatedSVD(
                n_components=self.vector_size,
                algorithm='randomized', 
                n_iter=10, 
                random_state=42
                )
            )
        )
        self.transformer = Pipeline(steps)

    def get_submodel(self, *mcodes):
        submodel = self.__class__(self.db.get_subdb(mcodes), self.tag, self.vocab_size, self.idf, self.vector_size, self.unit)
        if len(self.tagged_vectors):
            submodel.reload_tagged_vectors()
        submodel.transformer = self.transformer
        return submodel

class FastText(JstatutreeVectorModelBase):
    MODEL_TYPE_TAG = 'fasttext'
    def __init__(self, db, tag, vector_size=None, unit='XSentence', workers=None, wvmodel_path=None):
        self._wvmodel_path = wvmodel_path or None
        self._wvmodel = None
        self.workers = workers or cpu_count()
        self.db = db
        self.tag = tag
        if self.wvmodel is None:
            vector_size = vector_size or 300
        else:
            vector_size = vector_size or self.wvmodel.vector_size
        super().__init__(db, tag, vector_size, unit)
        
    def get_submodel(self, *mcodes):
        submodel = self.__class__(self.db.get_subdb(mcodes), self.tag, self.wvmodel_path, self.vector_size, self.unit, self.workers)
        submodel.reload_tagged_vectors()
        return submodel

    @property
    def wvmodel_path(self):
        return self._wvmodel_path or self.path/'fasttext'/'model.bin'

    @property
    def wvmodel(self):
        if self._wvmodel is None:
            try:
                self._wvmodel = gensim.models.FastText.load_fasttext_format(str(self.wvmodel_path))
            except:
                traceback.print_exc()
                print('wvmodel auto load failed', self.wvmodel_path)
                pass
        return self._wvmodel
    
    def save(self):
        with open(Path(self.path, 'jsmodel.pkl'), 'wb') as f:
            wvmodel = self.wvmodel
            self._wvmodel = None
            joblib.dump(self, f)
            self._wvmodel = wvmodel
    
    def _calc_vec(self, text):
        arr = np.array([self.wvmodel.wv[v] for v in text if v in self.wvmodel.wv])
        if arr.shape == np.array([]).shape:
            print("WARNING: a zero vector allocated for the text below:")
            print(text)
            return np.zeros(self.wvmodel.wv.vector_size)
        v = np.sum(arr, axis=0)
        return v/np.linalg.norm(v)
    
    def _vector_production(self, mcodes):
        import os
        import traceback
        from pprint import pprint
        np.seterr(all='raise')
        pid = os.getpid()
        db = self.db.get_subdb(mcodes)
        print('<proc {}>'.format(pid), 'task begin', db.mcodes)
        with self.vecs.get_subdb(mcodes).write_batch(transaction=True, sync=True) as wb:
            for tag, words in SentenceGenerator(db, self.unit, True):
                wv_sum = None
                try:
                    wv_sum = np.sum(np.array([self.wvmodel[w]for w in words if w in self.wvmodel.wv]), axis=0)
                    if wv_sum.shape != (self.vector_size,):
                        print('skip:', tag)
                    vector = wv_sum/np.linalg.norm(wv_sum)
                    wb.put(tag, vector)
                except FloatingPointError:
                    print('skip')
                    print(traceback.format_exc())
                    if wv_sum is not None:
                        print(wv_sum, type(wv_sum))
                        pprint(wv_sum)
            print('<proc {}>'.format(pid), "begin sync write")
        print('<proc {}>'.format(pid), 'task finished')
        return mcodes
    
    def __getstate__(self):
        self._wvmodel = None
        return self.__dict__    
    
    def fit(self, task_size=None): #task_size argument is just for compatibility
        if self.is_fitted():
            print('model already exists.')
            return
        print('Training begin')
        if self.wvmodel is None:
            os.makedirs(self.wvmodel_path.parent, exist_ok=True)
            print("training fasttext model")
            model = train_unsupervised(
                input=str(self.get_training_corpus()/'corpus.txt'),
                model='skipgram',
                dim = self.vector_size
            )
            model.save_model(str(self.wvmodel_path))
            del model
        
        target_mcodes = sorted(self.db.mcodes, key=lambda x: int(x))
        task_num = len(target_mcodes)
        finished_mcodes = []
        task_size = max(task_size or ( task_num + (self.workers-1))//self.workers, 1)

        print('mcode count:', task_num)
        print('task size: {} municipalities / proc'.format(task_size))
        print('proc count:', cpu_count())
        print('workers:', self.workers)
        with ProcessPoolExecutor(self.workers) as executor:
            futures = [executor.submit(self._vector_production, target_mcodes[i:i+task_size]) for i in range(0, len(target_mcodes), task_size) if not print('submit:',target_mcodes[i:i+task_size])]
            for done_mcodes in concurrent.futures.as_completed(futures):
                finished_mcodes +=done_mcodes.result()
                print('model construction progress: {0}%'.format(100*len(finished_mcodes)//task_num))
        assert len(target_mcodes)==len(finished_mcodes) and set(target_mcodes) == set(finished_mcodes)
        print('Model fitting complete!')
        self.save()
        
    def get_wmd_model(self):
        return FastTextWMD(self.db, self.tag, self.wvmodel_path, self.vector_size, self.unit)

class FastTextWMD(FastText):
    def _calc_vec(self, text):
        raise Exception("WMD model do not using vector")
    
    def get_wmd(self, tag1, tag2):
        t = time()
        wmd = round(
            self.wvmodel.wmdistance(
                [word for sentence in self.db.get_element(tag1).itersentence() for word in sentence], 
                [word for sentence in self.db.get_element(tag2).itersentence() for word in sentence]
            )
        )
        get_logger('FastTextWMD.get_wmd()').info('time: %.03f', time()-t)
        print(type(wmd), )
        return wmd
            
    def _vector_production(self, mcodes):
        raise Exception("WMD model do not using vector")

    def build_match_factory(self, query_key, theta, match_factory_cls, weight_border=0.5, sample_num=500):
        logger = get_logger(self.__class__.__name__+'.build_match_factory')
        query = self.db.get_jstatutree(query_key).change_root(query_key)
        match_factory = match_factory_cls(query, tree_factory=self.rspace_db.get_jstatutree)
        t = time()
        rankings = list(self.most_similar_by_keys([query_key], topn=sample_num, sample_num=sample_num))
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


    def most_similar_by_keys(self, keys, topn=10, name_as_tag=False, append_text=False, sample_num=None):
        if not sample_num or sample_num < topn:
            sample_num = topn*10
        ukeys, query_vectors = self.keys_to_vectors(*keys, return_keys=True)
        if not len(self.tagged_vectors):
            self.reload_tagged_vectors()
        ret = []
        rankings = [
            (
                qtag, 
                sorted([(t, self.get_wmd(qtag, t)) for t, d in ranking], key=lambda x: x[1])[:topn]
            )
                for qtag, ranking in zip(ukeys, self.tagged_vectors.knn(query_vectors, k=sample_num))
        ]
        for qtag, tag_dist_pairs in rankings:
            #print(tag_dist_pairs)
            ranking = tag_dist_pairs
            tag2sent = lambda x: ''.join([re.sub('▁', '', w) for s in self.db.get_element(x).itersentence() for w in s])
            if append_text:
                ranking = ((t, d, tag2sent(t)) for t, d in ranking)
            if name_as_tag:
                ranking = ((self.db.get_element_name(x[0], x[0]), *x[1:]) for x in ranking)
            ret.append(list(ranking))
        return zip(ukeys, ret)
    
class ReverseSentenceDB(replyvel.DB):
    @classmethod
    def sentence_hash(cls, x):
        return hashlib.md5(x.encode()).digest()
    
    @classmethod
    def _encode_key(cls, x):
        mcode, sent = x
        return mcode, cls.sentence_hash(sent)
        
    @classmethod
    def _decode_key(cls, mcode, sent):
        #print('decode:', mcode, sent)
        return (mcode, sent)
        
class SimString(JstatutreeModelCore):
    MODEL_TYPE_TAG = 'simstring'
    def __init__(self, db, tag, unit='XSentence', ngram=3, method=simstring.cosine, workers=None):
        assert not hasattr(db, 'tokenizer'), 'Normal JstatutreeDB required'
        super().__init__(db, tag, unit)
        self.method = method
        self.ngram = ngram
        self.workers = workers or cpu_count()
        self.reverse_unitdb = ReverseSentenceDB(Path(self.path, 'reverse_unitdb.rpl'), target_codes=self.db.target_codes)
        self.simstring_path = Path(self.path, 'simstring', '{}-gram'.format(ngram)).resolve()
        self.rspace_reversed_dict = None
        if self.simstring_path.exists():
            self.simstring = dict()
        else:
            self.simstring = None

    def build_match_factory(self, query_key, theta, match_factory_cls, weight_border=0.5):
        logger = get_logger(self.__class__.__name__+'.build_match_factory')
        query = self.db.get_jstatutree(query_key).change_root(query_key)
        if self.rspace_reversed_dict is None:
            self.restrict_rspace(self.db.mcodes)
        match_factory = match_factory_cls(query, tree_factory=self.rspace_db.get_jstatutree)
        simstrings = {mcode: simstring.reader(str(Path(self.simstring_path, mcode, 'db')))
                      for mcode in self.rspace_db.mcodes if not print('open:', str(Path(self.simstring_path, mcode, 'db')))}
        for ss in simstrings.values():
            ss.measure = self.method
            ss.threshold = theta
        usents = {uk: sent for uk, sent in query.iterXsentence(include_code=True, include_value=True)}
        def printer(val):
            print(val)
            return val
        entire_leaves_count = 0
        for i, (qkey, qsent) in enumerate(usents.items()):
            qnode = self.rspace_db.get_element(qkey)
            print(qnode.code)
            sims = np.array([(not match_factory.add_leaf_by_nodes(qnode, skey, 1.0))
                             for sent in set(s for ss in simstrings.values() for s in ss.retrieve(qsent))
                             for skey in self.rspace_reversed_dict.get(self.reverse_unitdb.sentence_hash(sent), []) if str(skey) not in str(query.lawdata.code)
                             #ifor sim in [self.calc_similarity(qsent, sent)] if sim >= theta
                             ])
            #query.find_by_code(qkey).attrib['weight'] = weight_border/(weight_border+np.sum(sims))
            logger.info('%03d/%d:add %d leaves similar to %s',i ,len(usents), len(sims), qkey)
            entire_leaves_count += len(sims)
            logger.info('Currently, %d trees stored in the builder.', len(match_factory.tree_store))
            if sims.shape[0] > 0:
                logger.info('Similarity summary: mean %.3f, max %.3f, min %.3f.', np.sum(sims)/max(sims.shape[0], 1), np.max(sims), np.min(sims))

        logger.info('Finally, add %d leaves', entire_leaves_count)
        return match_factory

    def policy_matching(self, keys, theta, activation_func, topn=10, weight_query_by_rescount=True, weight_border=5, **kwargs):
        match_factory = PolicyMatchFactory([self.db.get_element(k) for k in keys], theta, activation_func)
        if self.rspace_reversed_dict is None:
            self.restrict_rspace(self.db.mcodes)
        simstrings = {mcode: simstring.reader(str(Path(self.simstring_path, mcode, 'db'))) 
                for mcode in self.rspace_db.mcodes if not print('open:', str(Path(self.simstring_path, mcode, 'db')))}
        for ss in simstrings.values():
            ss.measure = self.method
            ss.threshold  = theta
        usents = {uk:''.join(self.db.get_element(uk).itersentence()) for uk in self.keys_to_ukeys(*keys)}
        rankings = []
        weights = {}
        for qkey, qsent in usents.items():
            """
            sims = []
            
            retrieved_sentences = set(s for gc, ss in simstrings.items() for s in ss.retrieve(qsent))
            if len(retrieved_sentences) == 0:
                continue
            print(qkey)
            print(qsent)
            print('{0} sents found'.format(len(retrieved_sentences)))
            retrieved_keys = []
            for sent in retrieved_sentences:
                keys = self.rspace_reversed_dict.get(self.reverse_unitdb.sentence_hash(sent), [])
                sim = self.calc_similarity(qsent, sent)
                sims.extend([sim]*len(keys))
                retrieved_keys.extend(keys)
                print(sent)
                minus = 0
                for skey in keys:
                    if skey == qkey:
                        minus = 1
                        continue
                        
                    match_factory.add_leaf(qkey, skey, sim, cip=1.0)
                print('{0} keys found, similarity is {1}'.format(len(keys)-minus, sim))
            print('Finally, {0} keys found '.format(len(retrieved_sentences)))
            """
            sims = np.array([(not match_factory.add_leaf(qkey, skey, sim, cip=1.0)) or sim
                    
                    for sent in set(s for ss in simstrings.values() for s in ss.retrieve(qsent))
                    for skey in self.rspace_reversed_dict.get(self.reverse_unitdb.sentence_hash(sent), []) if sum(1 if k in skey else 0 for k in keys)==0
                    for sim in [self.calc_similarity(qsent, sent)] if sim >= theta
                ])
            if weight_query_by_rescount:
                weights[qkey] = weight_border/(weight_border+np.sum(sims))
        match = match_factory.construct_matching_object(tree_factory = self.db.get_jstatutree)
        for qkey, w in weights.items():
            match.find_elem(qkey).attrib["weight"] = w
        return match
            
    @staticmethod
    def get_ngram_set(s, n):
        s = '▁'*(n-1) + s + '▁'*(n-1)
        ngrams = []
        for i in range(len(s)-(n-1)):
            ngram = s[i:i+n]
            while ngram in ngrams:
                ngram += '#'
            ngrams.append(ngram)
        return set(ngrams)
    
    def calc_similarity(self, x, y):
        if self.method == simstring.cosine:
            return round([ len(nx&ny) / math.sqrt( len(nx)*len(ny) )  for nx in [self.get_ngram_set(x, self.ngram)] for ny in [self.get_ngram_set(y, self.ngram)] ][0], 3)
        elif self.method == simstring.dice:
            return round([2*len(nx&ny) /( len(nx)+len(ny) )  for nx in [self.get_ngram_set(x, self.ngram)] for ny in [self.get_ngram_set(y, self.ngram)] ][0], 3)
    
    def _fit_unit_task(self, mcodes):
        pid = os.getpid()
        print('<proc {}>'.format(pid), 'task begin', mcodes)
        for mcode in mcodes:
            subsimstring_path = Path(self.simstring_path, mcode)
            subdb = self.reverse_unitdb.get_subdb([mcode])
            subsimstring = None
            if not subsimstring_path.exists():
                os.makedirs(str(subsimstring_path))
                subsimstring = simstring.writer(str(Path(subsimstring_path, 'db')))
            insert_dict = None
            if len(subdb) == 0:
                insert_dict = {}
            if subsimstring is None and insert_dict is None:
                print('<proc {}>'.format(pid), 'skip muni', mcode)
                continue
            for k, s in SentenceGenerator(self.db.get_subdb([mcode]), self.unit, include_key=True):
                if insert_dict is not None:
                    key = mcode+s
                    val = insert_dict.get(key, [])
                    val.append(k)
                    insert_dict[key] = val
                subsimstring is None or subsimstring.insert(s)
            if insert_dict is not None:
                with subdb.write_batch(sync=True, transaction=True) as wb:
                    for k, v in insert_dict.items():
                        wb.put((k[:6], k[6:]), v)
        return mcodes

    def restrict_rspace(self, rspace_codes):
        if self.rspace_reversed_dict is not None and set(self.rspace_codes) == set(rspace_codes):
            return
        self._rspace_changed_flag = True
        self.rspace_codes = rspace_codes
        self.rspace_reversed_dict = {}
        for mdb in self.reverse_unitdb.get_subdb(self.rspace_codes).split_unit():
            for (mcode, sentence_hash), code in mdb.iterator():
                val = self.rspace_reversed_dict.get(sentence_hash, None)
                if val is None:
                    val = code
                else:
                    val.extend(code)
                self.rspace_reversed_dict[sentence_hash] = val
    
    def fit(self, task_size=None):
        print('Training begin')

        target_mcodes = sorted(self.db.mcodes, key=lambda x: int(x))
        task_num = len(target_mcodes)
        finished_mcodes = []
        task_size = max(task_size or ( task_num + (self.workers-1))//self.workers, 1)
        
        print('mcode count:', task_num)
        print('task size: {} municipalities / proc'.format(task_size))
        print('proc count:', cpu_count())
        print('workers:', self.workers)

        with ProcessPoolExecutor(self.workers) as executor:
            futures = [executor.submit(self._fit_unit_task, target_mcodes[i:i+task_size]) for i in range(0, len(target_mcodes), task_size) if """not print('submit:',target_mcodes[i:i+task_size])"""]
            for done_mcodes in concurrent.futures.as_completed(futures):
                finished_mcodes +=done_mcodes.result()
                print('model construction progress: {0}%'.format(100*len(finished_mcodes)//task_num))
        print('Model fitting complete!')
        self.save()
    
    def most_similar_by_keys(self, keys, topn, init_threshold=0.7, **kwargs):
        if self.rspace_reversed_dict is None:
            self.restrict_rspace(self.db.mcodes)
        simstrings = {mcode: simstring.reader(str(Path(self.simstring_path, mcode, 'db'))) 
                for mcode in (self.rspace_codes or self.db.mcodes) if not print('open:', str(Path(self.simstring_path, mcode, 'db')))}
        for ss in simstrings.values():
            ss.measure = self.method
        usents = {uk:''.join(self.db.get_element(uk).itersentence()) for uk in self.keys_to_ukeys(*keys)}
        rankings = []
        for qkey, qsent in usents.items():
            results = []
            th = init_threshold
            while len(results) < topn and th >= 0:
                #print(qkey, th, len(results))
                ss.threshold = th
                results = [(skey, 1-sim)
                        for ss in simstrings.values()
                        for retrieved in [ss.retrieve(qsent)] if len(retrieved) >= topn
                        for sent in retrieved
                        for skey in self.rspace_reversed_dict.get(self.reverse_unitdb.sentence_hash(sent), [])
                        for sim in [self.calc_similarity(qsent, sent)]
                    ]
                th -= 0.05
            rankings.append([qkey, sorted(results, key=lambda x: -x[1])[:topn]])
        return rankings
