import os
import re
import shutil
import concurrent
import numpy as np
import pickle
import gensim

from pathlib import Path
from multiprocessing import cpu_count
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline

from .model_core import JstatutreeModelCore. JstatutreeVectorModelBase, ScikitModelBase
from .dataset import SentenceGenerator
from concurrent.futures import ProcessPoolExecutor
import simstring
import hashlib


class LSI(TfIdf):
    MODEL_TYPE_TAG = 'lsi'
    def __init__(self, tdb, tag, vocab_size=16000, idf=True, vector_size=300, unit='XSentence'):
        super().__init__(tdb, tag, vector_size, unit)
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
        self.steps.append((
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
        submodel.reload_tagged_vectors()
        submodel.transformer = self.transformer
        return submodel

class TfIdf(ScikitModelBase):
    MODEL_TYPE_TAG = 'tfidf'
    def __init__(self, tdb, tag, vocab_size=8000, idf=True, unit='XSentence'):
        super().__init__(tdb, tag, vocab_size, unit)
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
        submodel.reload_tagged_vectors()
        submodel.transformer = self.transformer
        return submodel


class FastText(JstatutreeModel):
    MODEL_TYPE_TAG = 'fasttext'
    def __init__(self, tdb, tag, wvmodel_path, vector_size=None, unit='XSentence', workers=None):
        self.wvmodel_path = wvmodel_path
        self._wvmodel = None
        self.workers = workers or cpu_count()
        vector_size = vector_size or self.wvmodel.vector_size
        super().__init__(tdb, tag, vector_size, unit)
        
    def get_submodel(self, *mcodes):
        submodel = self.__class__(self.db.get_subdb(mcodes), self.tag, self.wvmodel_path, self.vector_size, self.unit, self.workers)
        submodel.reload_tagged_vectors()
        return submodel
    
    def get_wmd(self, tag1, tag2):
        tag2sent = lambda tag: [word for sentence in self.db.get_element(tag).itersentence() for word in sentence]
        return round(self.wvmodel.wmdistance(tag2sent(tag1), tag2sent(tag2)), 3)
    
    def most_similar_by_keys(self, keys, topn=10, name_as_tag=False, append_text=False, append_olap=False, olap_measure='cosine', olap_ngram=3, using_wmd=False, wmd_topn=None):
        if not wmd_topn or wmd_topn < topn:
            wmd_topn = topn*10
        ukeys, query_vectors = self.keys_to_vectors(*keys, return_keys=True)
        olap_func = self.OLAP_FUNCTIONS.get(olap_measure, 'cosine')
        if not len(self.tagged_vectors):
            self.reload_tagged_vectors()
        ret = []
        if using_wmd:
            rankings = [
                (
                    qtag, 
                    sorted([(t, self.get_wmd(qtag, t)) for t, d in ranking], key=lambda x: x[1])[:topn]
                )
                    for qtag, ranking in zip(ukeys, self.tagged_vectors.knn(query_vectors, k=wmd_topn))
            ]
        else:
            rankings = zip(ukeys, self.tagged_vectors.knn(query_vectors, k=topn))
        for qtag, tag_dist_pairs in rankings:
            #print(tag_dist_pairs)
            ranking = tag_dist_pairs
            tag2sent = lambda x: ''.join([re.sub('▁', '', w) for s in self.db.get_element(x).itersentence() for w in s])
            if append_text:
                ranking = (
                    (t, d, tag2sent(t), olap_func(tag2sent(t), tag2sent(qtag), olap_ngram)) if append_olap else (t, d, tag2sent(t))
                      for t, d in ranking  
                )
            elif append_olap:
                ranking = (
                    (t, d, olap_func(tag2sent(t), tag2sent(qtag), olap_ngram))
                      for t, d in ranking  
                )
            if name_as_tag:
                ranking = ((self.db.get_element_name(x[0], x[0]), *x[1:]) for x in ranking)
            ret.append(list(ranking))
        return zip(ukeys, ret)
    
    @property
    def wvmodel(self):
        if self._wvmodel is None:
            self._wvmodel = gensim.models.FastText.load_fasttext_format(self.wvmodel_path)
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
                try:
                    wv_sum = np.sum(np.array([self.wvmodel[w]for w in words if w in self.wvmodel.wv]), axis=0)
                    if wv_sum.shape != (self.vector_size,):
                        print('skip:', tag)
                    vector = wv_sum/np.linalg.norm(wv_sum)
                    wb.put(tag, vector)
                except FloatingPointError:
                    print('skip')
                    print(traceback.format_exc())
                    print(wv_sum, type(wv_sum))
                    pprint(wv_sum)
            print('<proc {}>'.format(pid), "begin sync write")
        print('<proc {}>'.format(pid), 'task finished')
        return mcodes
    def __getstate__(self):
        self._wvmodel = None
        return self.__dict__

    def fit(self, task_size=None):
        self.reload_tagged_vectors()
        if len(self.tagged_vectors):
            print('model already exists.')
            return
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
            futures = [executor.submit(self._vector_production, target_mcodes[i:i+task_size]) for i in range(0, len(target_mcodes), task_size) if not print('submit:',target_mcodes[i:i+task_size])]
            for done_mcodes in concurrent.futures.as_completed(futures):
                finished_mcodes +=done_mcodes.result()
                print('model construction progress: {0}%'.format(100*len(finished_mcodes)//task_num))
        assert len(target_mcodes)==len(finished_mcodes) and set(target_mcodes) == set(finished_mcodes)
        print('Model fitting complete!')
        self.save()

class SimString(JstatutreeModelCore):
    def __init__(self, db, tag, unit='XSentence', ngram=3, method=simstring.cosine):
        assert hasattr(db, 'tokenizer'), 'TokenizedJstatutreeDB required'
        super().__init__(db, tag, unit)
        self.method = method
        self.reverse_unitdb = replyvel.DB(Path(self.path, 'reverse_unitdb.rpl'), target_codes=self.db.target_codes)
        self.reverse_unitdb._encode_key = lambda x: x
        self.reverse_unitdb._decode_key = lambda x: x
        self.simstring_path = Path(self.path, 'simstring', '{}-gram'.format(ngram)).resolve()
        if self.simstring_path.exists():
            self.simstring = dict()
        else:
            self.simstring = None

    hash_sentence = lambda x: hashlib.md5(x).digest()

    def _fit_unit_task(self, mcodes):
        pid = os.getpid()
        print('<proc {}>'.format(pid), 'task begin', db.mcodes)
        for mcode in mcodes
            subsimstring_path = Path(self.simstring_path, mcode)
            subdb = self.reverse_unitdb.get_subdb([mcode])
            subsimstring = None
            if not subsimstring_path.exists():
                os.path.mkdirs(subsimstring_path)
                subsimstring = simstring.writer(str(Path(subsimstring_path, 'db')))
            insert_dict = None
            if len(subdb) = 0:
                insert_dict = {}
            if subsimstring is None and insert_dict is None:
                print('<proc {}>'.format(pid), 'skip muni', mcode)
                continue
            for k, s in SentenceGenerator(self.db.get_subdb([mcode]), self.unit, include_key=True):
                if insert_dict is not None:
                    hs = self.hash_sentence(s)
                    val = self.insert_dict.get(hs, [])
                    val.append(k)
                    self.insert_dict[hs] = val
                subsimstring is None or subsimstring.insert(s)
            if insert_dict is not None:
                with subdb.write_batch as wb:
                    for k, v in insert_dict.values():
                        wb.put(k, v)
        return mcodes

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
            futures = [executor.submit(self._fit_unit_task, target_mcodes[i:i+task_size]) for i in range(0, len(target_mcodes), task_size) if not print('submit:',target_mcodes[i:i+task_size])]
            for done_mcodes in concurrent.futures.as_completed(futures):
                finished_mcodes +=done_mcodes.result()
                print('model construction progress: {0}%'.format(100*len(finished_mcodes)//task_num))
        print('Model fitting complete!')
        self.save()
    
    def most_similar_by_keys(self, keys, topn, init_threshold=0.7):
        simstrings = {mcode: simstring.reader(str(Path(self.simstring_path, mcode))) 
                for mcode in self.rspace_codes or self.db.mcodes}
        for ss in simstrings.values():
            ss.measure = self.method
        usents = {uk:''.join(self.db.get_element(uk)).itersentence() for uk in self.keys_to_ukeys(keys)}
        rankings = []
        for qkeys, qsents in usents.items():
            results = []
            th = init_threshold
            while len(results) < topn and th >= 0
                ss.threshold = th
                results = [(skey, sim)
                        for ss in simstrings.values()
                        for sent in ss.retrieve(qsent)
                        for skey in self.reverse_unitdb.get(sent, [])
                        for sim in [self.calc_similarity(qsent, sent)] if sim > threshold
                    ]
                th -= 0.05
            rankings.append(qtag, sorted(results, key=lambda x: -x[1])[:topn])
        return rankings
