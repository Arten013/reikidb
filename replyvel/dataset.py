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


class SentenceGenerator(object):
    def __init__(self, db, unit, include_key=False):
        self.db = db
        self.unit = unit
        self.include_key = include_key

    def export_corpus(self, path, workers=None):
        # todo: complete this function
        workers = workers or cpu_count()
        with concurrent.futures.ProcessPoolExecutor(workers) as executor:
            vectors = np.vstack([v for t, v in executor.map(self._vector_production, [db for c, db in self.db.split_by_dbcount(self.workers)]   ) if not tags.extend(t)])
    
    def __iter__(self):
        if self.unit == 'XSentence':
            sg = ((k, v) for t in self.db.iter_jstatutree(include_tag=False) for k, v in t.iterXsentence(include_code=True) if v is not None and len(v) > 0)
        else:
            sg =  ((k, v) for k, v in self.db.iter_element_sentences(self.unit, include_tag=True) if v is not None and len(v) > 0)
        if self.include_key:
            yield from sg
        else:
            yield from (v for k, v in sg)
    
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
        return TokenizedJstatutreeDB(self.path, tokenizer, self.target_codes)
    
    def remove_files(self):
        if self.path.exists():
            print('rmdir:', self.path)
            shutil.rmtree(self.path)
    
    def get_subdb(self, target_codes):
        for code in target_codes:
            if not self.jstatutree_db._code_ptn_match(code):
                raise ValueError('Invalid code: '+str(code))
        return self.__class__(self.path, target_codes)
    
    def split_by_pref(self):
        return [ (pcode, self.get_subdb(list(mcodes))) for pcode, mcodes in groupby(sorted(self.mcodes, key=lambda x: int(x)), key=lambda x: x[:2])]
    
    def split_by_muni(self):
        return [(mcode, self.get_subdb([mcode])) for mcode in self.mcodes]
    
    def split_by_dbcount(self, dbcount):
        db_size = ( len(list(self.mcodes)) + (dbcount-1))//dbcount
        return self.split_by_dbsize(dbsize)

    def split_by_dbsize(self, dbsize):
        entire_mcodes = sorted(self.mcodes, key=lambda x: int(x))
        step_size = dbsize
        return [(mcodes, self.get_subdb(mcodes))
                    for i in range(0, len(entire_mcodes), step_size)
                    for mcodes in [ entire_mcodes[i:i+step_size] ]
                ]
    
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
     
        
    def get(self, code, default=None):
        parts = Path(code).parts
        if len(parts) == 3:
            return self.get_jstatutree(code, default)
        elif len(parts) > 3:
             if code2etype(code) in self.element_db:
                return self.get_element(code, default)
        return default
    
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
        if code2etype(code) not in self.element_db:
            ret = self.get_jstatutree(code, None)
            return None if ret is None else ret.getroot()
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
    
    def iter_lawcodes(self, target_codes=None):
        yield from self.jstatutree_db.iterator(include_key=True, include_value=False, target_codes=target_codes)

    def __len__(self):
        return sum(1 for v in self.iter_lawcodes())
    
    def iter_elements(self, target_etype, include_tag=True):
        for elem in self.element_db[target_etype].iterator(include_key=False, include_value=True):
            yield self._complete_element(elem)
            
    def iter_elements(self, target_etype, include_tag=True):
        #if target_etype == 'Xsentence':
        #    yield from self.iter_Xsentence(self, )
        for elem in self.element_db[target_etype].iterator(include_key=False, include_value=True):
            yield self._complete_element(elem)
            
    def iter_element_sentences(self, target_etype, include_tag=True):
        for elem in self.iter_elements(target_etype, False):
            ret = ''
            for s in elem.itersentence():
                ret += self.sentence_db.get(s.code)
            yield (elem.code, ret) if include_tag else ret
            
    def iter_jstatutree(self, include_tag=True, include_value=True, target_codes=None):
        if not include_value:
            return self.iter_lawcodes(target_codes=target_codes)
        for jst in self.jstatutree_db.iterator(include_key=False, include_value=True, target_codes=target_codes):
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
                if re.match('\.DS_Store$', str(path)):
                    continue
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
                except Exception as e:
                    raise e

class MultiProcWriter(object):
    def __init__(self, db_path=None, db=None, target_codes='ALL', workers=None, tokenizer=None):
        if db is not None:
            self.db = db
        elif db_path is not None:
            if tokenizer:
                self.db = TokenizedJstatutreeDB(path=db_path, tokenizer=tokenizer, target_codes=target_codes)
            else:
                self.db = JstatutreeDB(path=db_path, target_codes=target_codes)
        else:
            raise Exception("You mast pass db or db_pass for initialize MultiProcWriter")
        self.workers=workers or cpu_count()
        
    def write_batch_pref_set(self, prefs_path, print_log=False, error_detail=True, print_skip=True, only_reiki=True, *wbargs, **wbkwargs):
        for pref_path in Path(prefs_path).iterdir():
            if pref_path.is_dir():
                self.write_batch_pref_directory(pref_path, print_log, error_detail, print_skip, only_reiki, *wbargs, **wbkwargs)
            
    def write_batch_pref_directory(self, pref_path, print_log=False, error_detail=True, print_skip=True, only_reiki=True, *wbargs, **wbkwargs):
        with concurrent.futures.ProcessPoolExecutor(self.workers) as executor:
            futures = []
            for muni_path in Path(pref_path).iterdir():
                if not muni_path.is_dir():
                    print('skip muni (not dir):',muni_path)
                mcode = Path(muni_path).name
                try:
                    subdb = self.db.get_subdb( [mcode])
                    print('submit path:', muni_path)
                    futures.append(
                        executor.submit(
                            subdb.write_batch_muni,
                            muni_path, 
                            print_log,
                            error_detail,
                            print_skip,
                            only_reiki, 
                            *wbargs,
                            **wbkwargs
                        )
                    )
                    if len(futures) > 30:
                        while len(futures) > 20:
                            _, futures = concurrent.futures.wait(futures, return_when=concurrent.futures.FIRST_COMPLETED)
                        futures = list(futures)
                except ValueError:
                    pass
                except:
                    raise
            concurrent.futures.wait(futures)
            
    def tokenize_batch(self, *wbargs, **wbkwargs):
        assert isinstance(self.db, TokenizedJstatutreeDB), 'No tokenizer'
        executor = concurrent.futures.ProcessPoolExecutor(self.workers)
        futures = []
        for mcode, muni_db in self.db.split_by_muni():
            print('submit mcode:', mcode)
            futures.append(executor.submit(muni_db.tokenize_batch, *wbargs, **wbkwargs))
            if len(futures) > 30:
                print('wait for finishing tasks ...')
                executor.shutdown(wait=True)
                executor = concurrent.futures.ProcessPoolExecutor(self.workers)
                print('restart')
                futures = []
        executor.shutdown(wait=True)

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
        return self.__class__(self.path, self.tokenizer, target_codes)
    
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
    
    #def iter_elements(self, target_etype, include_tag=True):
    #    for elem in self.element_db[target_etype].iterator(include_key=False, include_value=True):
    #        yield self._complete_element(elem)
    
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
    
        
