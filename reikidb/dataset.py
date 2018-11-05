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

from . import replyvel

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline
from sklearn.neighbors import NearestNeighbors
import numpy as np
import pickle

class JstatutreeLSI(object):
    def __init__(self, tdb, tag, vocab_size=16000, idf=True, size=500, unit='XSentence'):
        assert hasattr(tdb, 'tokenizer'), 'TokenizedJstatutreeDB required'
        self.db = tdb
        self.tag = tag
        self.vecs = replyvel.DB(self.path, target_codes=self.db.target_codes)
        self.vocab_size = vocab_size
        self.idf = idf
        self.size = size
        self.unit = unit
        self.transformer = self.construct_transformer()
        self.tag_idx_dict = {}
        self.tags = []
        self.matrix = None
    
    def remove_files(self):
        shutil.rmtree(self.path)
    
    def construct_transformer(self):
        count_vectorizer = CountVectorizer(
            input='content',
            #max_df=0.5, 
            #min_df=1, 
            lowercase = False,
            max_features=self.vocab_size
            )
        steps = [('CountVectorize', count_vectorizer)]
        if self.idf:
            steps.append(("TfidfTransform", TfidfTransformer()))
        steps.append(
                    ( 
                        "TruncatedSVD",
                        TruncatedSVD(n_components=self.size, algorithm='randomized', n_iter=10, random_state=42)
                    )
        )
        return Pipeline(steps)
    
    def load_matrix(self):
        with open(Path(self.path, 'matrix.npy'), "rb") as f:
            self.matrix = np.load(f)
        with open(Path(self.path, 'tags.pkl'), "rb") as f:
            self.tags = pickle.load(f)
        with open(Path(self.path, 'tag_idx_dict.pkl'), "rb") as f:
            self.tag_idx_dict = pickle.load(f)
    
    def most_similar(self, query_matrix, topn=10, name_as_tag=True, append_text=False):
        if self.matrix is None:
            self.load_matrix()
        distances_array, indice_array = NearestNeighbors(n_neighbors=topn, metric='cosine', algorithm='brute').fit(self.matrix).kneighbors(query_matrix)
        ret = []
        for dl, il in zip(distances_array.tolist(), indice_array.tolist()):
            dl = [round(d, 3) for d in dl]
            item = zip([self.tags[i] for i in il], [round(d, 3) for d in dl])
            if append_text:
                item = (
                    (t, d, ''.join([w for s in self.db.get_element(t).itersentence() for w in s]))
                    for t, d in item)
            if name_as_tag:
                item = ((self.db.get_element_name(x[0], x[0]), *x[1:]) for x in item)
            ret.append(list(item))
        return ret
    
    def most_similar_by_keys(self, keys, *args, **kwargs):
        ukeys = []
        for key in (keys if isinstance(keys, list) else [keys]):
            print(key)
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
        return self.most_similar(np.matrix([self.vecs.get(k) for k in ukeys]), *args, **kwargs)

    def reg_matrix_idx(self, i, key, words):
        self.tag_idx_dict[key] = i
        self.tags.append(key)
        return words
    
    def _calc_vec(self, text):
        if isinstance(text, str):
            text = self.db.tokenizer(text)
        return self.transformer.transform(text)
    
    def fit(self):
        if Path(self.path, 'matrix.npy').exists():
            print('model already exists.')
            return
        if self.unit == 'XSentence':
            sg = ((k, v) for t in self.db.iter_jstatutree(include_tag=False) for k, v in t.iterXsentence(include_code=True) if v is not None and len(v) > 0)
        else:
            sg = self.db.iter_element_sentences(self.unit, include_tag=True)
        self.matrix = self.transformer.fit_transform(
            (
                " ".join(self.reg_matrix_idx(i, key, words))
                for i, (key, words) in enumerate(sg)
            )
        )
        with self.vecs.write_batch() as wb:
            for tag in self.tags:
                wb.put(tag, self.matrix[self.tag_idx_dict[tag]])
        with open(Path(self.path, 'matrix.npy'), "wb") as f:
            np.save(f, self.matrix)
        with open(Path(self.path, 'tags.pkl'), "wb") as f:
            pickle.dump(self.tags, f)
        with open(Path(self.path, 'tag_idx_dict.pkl'), "wb") as f:
            pickle.dump(self.tag_idx_dict, f)
        
    @property
    def path(self):
        return Path(self.db.path, 'tfidf', self.tag)
    
    def get(self, key, default=None):
        ret = self.vecs.get(key, None)
        if ret is not None:
            return ret
        e = self.db.get_element(key, None)
        if e is None:
            return default
        words = [w for s in e.itersentences() for w in s]
        return self._calc_vec(text)
    
class JstatutreeDB(object):
    def __init__(self, path, target_codes='ALL'):
        self.path = Path(path)
        self.target_codes = target_codes
        self.jstatutree_db = replyvel.DB(Path(self.path, 'jstatutree'), target_codes=self.target_codes) 
        self.element_db = {e.__name__: replyvel.DB(Path(self.path, "elements", e.__name__), target_codes=self.target_codes) for e in ETYPES}
    
    @property
    def mcodes(self):
        return self.jstatutree_db.mcodes
    
    def get_tokenized_db(self, tokenizer):
        return TokenizedJstatureeDB(self.path, tokenizer, self.target_codes)
    
    def remove_files(self):
        shutil.rmtree(self.path)
    
    def get_subdb(self, target_codes):
        return JstatutreeDB(self.path, target_codes)
    
    def split_by_pref(self):
        return [ (pcode, self.get_subdb(mcodes)) for pcode, mcodes in groupby(self.mcodes, lambda x: x[:2])]
    
    def split_by_muni(self):
        return [(mcode, self.get_subdb([mcode])) for mcode in self.mcodes]
    
    def get_analyzer(self, workers=None):
        return JstatutreeDBAnalyzer(self, workers)
    
    def put_jstatutree(self, jstree):
        root = jstree.getroot()
        self.put_element(root)
        jstree._root = root.code
        self.jstatutree_db.put(str(jstree.lawdata.code), jstree)
        
    def put_element(self, element):
        assert isinstance(element, Element)
        for child in list(element):
            self.put_element(child)
        self.element_db[element.etype].put(element.code, element)
     
    def get_jstatutree(self, code, default=None):
        code = str(ReikiCode(code))
        jstree = self.jstatutree_db.get(code, None)
        if not jstree:
            return default
        jstree._root = self.get_element(jstree._root, [])
        return jstree
    
    def get_element(self, code, default=None):
        code = str(code)
        etype = code2etype(code)
        elem = self.element_db[etype].get(code)
        if elem is None:
            return default
        return self._complete_element(elem)
    
    def get_element_name(self, code, default=None):
        etype = code2etype(str(code))
        lawcode = ReikiCode(code)
        lawname = self.jstatutree_db.get(str(lawcode)).lawdata.name
        return lawname+code2jname(str(code))
        
    def _complete_element(self, elem):
        elem._children = [self.get_element(child_code, None) for child_code in list(elem)]
        return elem
    
    def iter_lawcodes(self):
        yield from self.jstatutree_db.iterator(include_key=True, include_value=False)

    def __len__(self):
        return sum(1 for v in self.iter_lawcodes())
 
    def iter_elements(self, target_etype, include_tag=True):
        for elem in self.element_db[target_etype].iterator(include_key=False, include_value=True):
            yield self._complete_element(elem)
            
    def iter_element_sentences(self, target_etype, include_tag=True):
        for elem in self.iter_elements(target_etype, False):
            ret = ''
            for s in elem.itersentence():
                ret += self.sentence_db.get(s.code)
            yield (elem.code, ret) if include_tag else ret
            
    def iter_jstatutree(self, include_tag=True, include_value=True):
        if not include_value:
            return self.iter_lawcodes()
        for jst in self.jstatutree_db.iterator(include_key=False, include_value=True):
            jst._root = self.get_element(jst._root)
            yield (str(jst.lawdata.code), jst) if include_tag else jst
    
    def write_batch(self, *args, **kwargs):
        return JstatutreeBatchWriter(self, *args, **kwargs)
    
    def write_batch_muni(self, path, print_log=False, error_detail=True, print_skip=False, only_reiki=True, *wbargs, **wbkwargs):
        path = Path(path)
        mcode = path.name
        assert re.match('\d{6}', mcode), 'Inappropriate path: '+str(path)
        with self.write_batch(*wbargs, **wbkwargs) as wb:
            for path in Path(path).iterdir():
                try:
                    jstree = Jstatutree(path)
                    if (not only_reiki) or jstree.lawdata.is_reiki():
                        wb.put_jstatutree(jstree)
                        print_log and print('add:', path, jstree.lawdata.name)
                    else:
                        (print_log or print_skip) and print('skip (Not Reiki):', path, jstree.lawdata.lawnum)
                except LawError as e:
                    if print_log or error_detail:
                        print('skip (Structure Error):', path)
                        if error_detail:
                            print(e)
                except ET.ParseError as e:
                    print_log and print('skip (Parse Error):', path)

class MultiProcWriter(object):
    def __init__(self, db_path, target_codes='ALL', workers=None, tokenizer=None):
        if tokenizer:
            self.db = TokenizedJstatutreeDB(db_path, tokenizer, target_codes)
        else:
            self.db = JstatutreeDB(db_path, target_codes)
        self.workers=workers or cpu_count()
        
    def write_batch_pref_set(self, prefs_path, print_log=False, error_detail=True, print_skip=True, only_reiki=True, *wbargs, **wbkwargs):
        for pref_path in Path(prefs_path).iterdir():
            self.write_batch_pref_directory(pref_path, print_log, error_detail, print_skip, only_reiki, *wbargs, **wbkwargs)
            
    def write_batch_pref_directory(self, pref_path, print_log=False, error_detail=True, print_skip=True, only_reiki=True, *wbargs, **wbkwargs):
        with concurrent.futures.ProcessPoolExecutor(self.workers) as executor:
            futures = []
            for muni_path in Path(pref_path).iterdir():
                print('submit path:', muni_path)
                futures.append(
                    executor.submit(
                        self.db.get_subdb( [muni_path.name] ).write_batch_muni,
                        muni_path, 
                        print_log,
                        error_detail,
                        print_skip,
                        only_reiki, 
                        *wbargs,
                        **wbkwargs
                    )
                )
            concurrent.futures.wait(futures)
            
    def tokenize_batch(self, *wbargs, **wbkwargs):
        assert isinstance(self.db, TokenizedJstatutreeDB), 'No tokenizer'
        workers = workers or cpu_count()
        with concurrent.futures.ProcessPoolExecutor(self.workers) as executor:
            futures = []
            for muni_db in self.db.split_by_muni():
                print('submit mcode:', list(muni_db.mcodes)[0])
                futures.append(executor.submit(muni_db.tokenize_batch, *wbargs, **wbkwargs))
            concurrent.futures.wait(futures)

import pandas as pd
class JstatutreeDBAnalyzer(object):
    def __init__(self, db, workers=None):
        self.db = db
        self.workers = workers
        
    def element_count(self, ignore_zero=True):
        df = pd.DataFrame(columns=["count"], )
        with concurrent.futures.ProcessPoolExecutor(self.workers) as executor:
            futures = {}
            for ename, edb in self.db.element_db.items():
                futures[ename] = executor.submit(len, edb)
            concurrent.futures.wait(futures.values())
            for ename in [e.__name__ for e in ETYPES]:
                value = futures[ename].result()
                if ignore_zero and value == 0:
                    continue
                df.loc[ename, "count"] = value
        return df.astype("int32")
    
    def text_keywords_search(self, keywords, etype=None, include_value=True):
        generator = self.db.iter_element_sentences(etype, True, True) if etype else self.db.iter_tokenized_sentences(include_tag=True)
        for tag, words in generator:
            for kw in keywords:
                if kw not in text:
                    yield (tag, words) if include_value else tag
     
    def name_keywords_search(self, keywords):
        for tag, jstree in self.jstatutree_db.iterator(include_key=True, include_value=True):
            for kw in keywords:
                if kw not in jstree.lawdata.name:
                    yield tag
    
class TokenizedJstatutreeDB(JstatutreeDB):
    def __init__(self, path, tokenizer, target_codes='ALL'):
        super().__init__(path, target_codes)
        self.tokenizer = tokenizer
        self.sentence_db = replyvel.DB(Path(self.path, "texts", self.tokenizer.tag, "Sentence"), target_codes=self.target_codes) 
        
    def get_normal_db(self):
        return JstatutreeDB(self.path, self.target_codes)
        
    def tokenize_batch(self, overwrite=False, *args, **kwargs):
        with self.sentence_db.write_batch(*args, **kwargs) as wb:
            for e in self.iter_elements('Sentence'):
                if e.text is None:
                    wb.put(e.code, [])
                if overwrite or isinstance(e.text, str):
                    wb.put(e.code, self.tokenizer.tokenize(e.text))
            
    def put_element(self, element):
        assert isinstance(element, Element)
        if element.etype == 'Sentence':
            self.sentence_db.put(element.code, self.tokenizer.tokenize(''.join(element.itertext())))
        for child in list(element):
            self.put_element(child)
        self.element_db[element.etype].put(element.code, element)
    
    def get_subdb(self, target_codes):
        return TokenizedJstatutreeDB(self.path, self.tokenizer, target_codes)
    
    def get_jstatutree(self, code, default=None):
        code = str(ReikiCode(code))
        jstree = self.jstatutree_db.get(code, None)
        if not jstree:
            return default
        jstree._root = self.get_element(jstree._root, [])
        return jstree
    
    def _complete_element(self, elem):
        elem._children = [self.get_element(child_code, None) for child_code in list(elem)]
        if elem.etype == 'Sentence':
            elem.text = self.sentence_db.get(elem.code, elem.text)
        return elem
    
    def iter_elements(self, target_etype, include_tag=True):
        for elem in self.element_db[target_etype].iterator(include_key=False, include_value=True):
            yield self._complete_element(elem)
    
    def iter_element_sentences(self, target_etype, include_tag=True):
        for elem in self.iter_elements(target_etype, False):
            ret = [w for s in elem.itersentence() for w in s]
            yield (elem.code, ret) if include_tag else ret

class JstatutreeBatchWriter(object):
    def __init__(self, jstatutree_db, *args, **kwargs):
        self.db = jstatutree_db
        self.tokenizer = self.db.tokenizer if hasattr(self.db, 'tokenizer') else None
        self.put_jstatutree = self.db.put_jstatutree
        self.put_element = self.db.put_element 
        self.jstatutree_db = self.db.jstatutree_db.write_batch(*args, **kwargs)
        self.element_db = {k: self.db.element_db[k].write_batch(*args, **kwargs) for k, v in self.db.element_db.items()}
        self.sentence_db = self.db.sentence_db.write_batch(*args, **kwargs) if hasattr(self.db, 'sentence_db') else None
    
    def delete(self, key):
        raise Exception('Not implemented yet')

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is not None:
            return False
        self.write()
        return True
    
    def write(self):
        self.jstatutree_db.write()
        for v in self.element_db.values():
            v.write()
        self.sentence_db and self.sentence_db.write()
