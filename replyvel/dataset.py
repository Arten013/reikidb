import os
import re
import shutil
import concurrent
from itertools import groupby
from pathlib import Path
from xml.etree import ElementTree as ET, ElementPath
import traceback

from jstatutree import Jstatutree
from jstatutree.exceptions import LawError
from jstatutree.element import Element
from jstatutree.etypes import ETYPES, code2etype, code2jname, CATEGORY_TEXT
from jstatutree.lawdata import ReikiCode
from multiprocessing import cpu_count

from . import _replyvel as replyvel
from .utils.logger import get_logger
from pygtrie import StringTrie
from time import time


class SentenceGenerator(object):
    def __init__(self, db, unit, include_key=False):
        self.db = db
        self.unit = unit
        self.include_key = include_key
    
    def sentence_list(self):
        return list(self)
    
    def __iter__(self):
        if self.unit == 'XSentence':
            sg = ((k, v) for t in self.db.iter_jstatutree(include_tag=False) for k, v in t.getroot().iterXsentence(include_code=True) if v is not None and len(v) > 0)
        elif self.unit == 'Sentence':
            sg = ((k, v) for t in self.db.iter_jstatutree(include_tag=False) for k, v in t.getroot().itersentence(include_code=True) if v is not None and len(v) > 0)
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
        self._preloaded_elements = StringTrie()
        self._preloaded_codes = []

    @property
    def mcodes(self):
        return self.jstatutree_db.mcodes
    
    def get_tokenized_db(self, tokenizer):
        return TokenizedJstatutreeDB(self.path, tokenizer, self.target_codes)
    
    def export_corpus(self, unit, path=None, workers=None, batch_size=30):
        corpus_path = Path((path or self.path)/'corpus.txt')
        tag_path = Path((path or self.path)/'tags.txt')
        if tag_path.exists():
            print(tag_path, "already exists.")
            return

        splitted_dbs = [d for c, d in self.split_by_muni()]
        
        with open(corpus_path, 'w') as sf:
            with open(tag_path, 'w') as tf:
                for db_batch in [splitted_dbs[i:i+batch_size] for i in range(0, len(splitted_dbs), batch_size)]:
                    with concurrent.futures.ProcessPoolExecutor(workers) as executor:
                        futures = [executor.submit(SentenceGenerator(subdb, unit, True).sentence_list) for subdb in db_batch]
                        for future in concurrent.futures.as_completed(futures):
                            [(sf.writelines(sent.rstrip()+'\n') , tf.writelines(tag+'\n')) for tag, sent in future.result()]
    
    def remove_files(self):
        if self.path.exists():
            print('rmdir:', self.path)
            shutil.rmtree(self.path)
    
    def get_subdb(self, target_codes):
        subdb=self.jstatutree_db.get_subdb(target_codes)
        return self.__class__(self.path, subdb.target_codes)
    
    def split_by_pref(self):
        return [ (pcode, self.get_subdb(list(mcodes))) for pcode, mcodes in groupby(sorted(self.mcodes, key=lambda x: int(x)), key=lambda x: x[:2])]
    
    def split_by_muni(self):
        return [(mcode, self.get_subdb([mcode])) for mcode in self.mcodes]
    
    def split_by_dbcount(self, dbcount):
        db_size = ( len(list(self.mcodes)) + (dbcount-1))//dbcount
        return self.split_by_dbsize(db_size)

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
        # jstree._root = root.code
        self.jstatutree_db.put(str(jstree.lawdata.code), jstree)
        
    def put_element(self, element):
        assert isinstance(element, Element)
        for child in list(element):
            self.put_element(child)
        self.element_db[element.etype].put(element.code, element)
     
    def __getitem__(self, key):
        val = self.get(key, None)
        if val is None:
            raise KeyError(key)
        return val
    
    def get(self, code, default=None):
        parts = Path(code).parts
        if len(parts) == 3:
            return self.get_jstatutree(code, default)
        elif len(parts) > 3:
            if code2etype(code) in self.element_db:
                return self.get_element(code, default)
            print("Warning: Unrecognizable object accessed from the function get_element()")
            print("Key:", code, "recognized as etype", code2etype(code))
        return default
    
    def get_jstatutree(self, code, default=None):
        t = time()
        code = str(ReikiCode(code))
        jstree = self.jstatutree_db.get(code, None)
        if not jstree:
            return default
        #jstree._root = self.get_element(jstree._root, None)
        get_logger('dataset.JstatutreeDB.get_jstatutree()').debug('time: %f sec', round(time()-t, 3))
        if jstree._root is None:
            return default
        return jstree

    PRELOAD_SIZE = 1000000
    def load_element(self, code:str):
        code = str(code)
        if code[:14] not in self._preloaded_codes:
            if len(self._preloaded_elements) > self.PRELOAD_SIZE:
                get_logger('dataset.JstatutreeDB.load_element()').info('discard _preloaded_elements(size: %d)', len(self._preloaded_elements))
                self._preloaded_elements = StringTrie()
                self._preloaded_codes = []
            self._preloaded_codes.append(code[:14])
            for etype in ['Sentence', 'Article', 'Paragraph', 'ParagraphSentence', 'Column']:
                # etype = etype.__name__
                #print(etype)
                for cc, ce in self.element_db[etype].iterator(include_key=True, include_value=True, prefix=code[:14]):
                    self._preloaded_elements[str(cc)] = ce
                    get_logger('').debug('preload: '+repr(ce))
        ret = self._preloaded_elements.get(code, None)
        if ret is None:
           ret = self.element_db[code2etype(code)].get(code, None)
        else:
            get_logger('dataset.JstatutreeDB.load_element()').debug('use preloaded: '+code)
        return ret

    def get_element(self, code, default=None, *, _use_preload=False):
        code = str(code)
        etype = code2etype(code)
        if etype not in self.element_db:
            print("Warning: Non-element object accessed from the function get_element()")
            print("Key:", code, "recognized as etype", etype)
            ret = self.get_jstatutree(code, None)
            return None if ret is None else ret.getroot()
        if _use_preload:
            elem = self.load_element(code)
        else:
            elem = self.element_db[etype].get(code, None)
        if elem is None:
            return default
        return self._complete_element(elem)
    
    def get_element_name(self, code, default=None):
        lawcode = ReikiCode(code)
        lawname = self.jstatutree_db.get(str(lawcode)).lawdata.name
        return lawname+code2jname(str(code))
        
    def _complete_element(self, elem):
        return elem
        if len(elem):
            elem._children = [self[child_code] for child_code in elem]
            assert None not in elem._children, 'Incomplete Element: ' + str(elem.code) + "\n" + str(list(elem)) 
        return elem
    
    def iter_lawcodes(self, target_codes=None):
        yield from self.jstatutree_db.iterator(include_key=True, include_value=False, target_codes=target_codes)

    def __len__(self):
        return sum(1 for v in self.iter_lawcodes())
    
    def iter_elements(self, target_etype, include_tag=True):
        for elem in self.element_db[target_etype].iterator(include_key=False, include_value=True):
            yield self._complete_element(elem)
            
    def iter_elements(self, target_etype, include_tag=True):
        # if target_etype == 'Xsentence':
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
        t = time()
        for jst in self.jstatutree_db.iterator(include_key=False, include_value=True, target_codes=target_codes):
            get_logger('dataset.JstatutreeDB.iter_jstatutree()').debug('time: %f sec', round(time()-t, 3))
            # jst._root = self.get_element(jst._root)
            yield (str(jst.lawdata.code), jst) if include_tag else jst
            t = time()
    
    def write_batch(self, *args, **kwargs):
        return JstatutreeBatchWriter(self, *args, **kwargs)
    
    def write_batch_muni(self, path, print_log=False, error_detail=True, print_skip=False, only_reiki=True, *wbargs, **wbkwargs):
        logger = get_logger('<proc:{1}>{0}write_batch_muni'.format(str(self.__class__), str(os.getpid())))
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
                        logger.debug('add: %s %s', path, jstree.lawdata.name)
                    else:
                        logger.debug('skip (Not Reiki): %s %s', path, jstree.lawdata.lawnum)
                except LawError as e:
                    if print_log or error_detail:
                        logger.debug('skip (Structure Error):', path)
                        logger.debug('error detail: %s', str(e))
                except ET.ParseError as e:
                    logger.warning('skip (Parse Error):', path)
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
        logger = get_logger('MultiProcWriter.write_batch_pref_directory')
        with concurrent.futures.ProcessPoolExecutor(self.workers) as executor:
            futures = []
            for muni_path in Path(pref_path).iterdir():
                if not muni_path.is_dir():
                    print('skip muni (not dir):',muni_path)
                mcode = Path(muni_path).name
                if not self.db.jstatutree_db._code_ptn_match(mcode):
                    continue
                try:
                    subdb = self.db.get_subdb( [mcode])
                    logger.info('submit: %s', str(mcode))
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
                        logger.info('wait task finishing')
                        while len(futures) > 20:
                            _, futures = concurrent.futures.wait(futures, return_when=concurrent.futures.FIRST_COMPLETED)
                        futures = list(futures)
                        logger.info('restart')
                except ValueError:
                    print(traceback.format_exc())
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

def deprecate(func):
    def wrapper(*args, **kwargs):
        raise Exception('Deprecated.')

class TokenizedJstatutreeDB(JstatutreeDB):
    def __init__(self, path, tokenizer, target_codes='ALL'):
        super().__init__(path, target_codes)
        self.tokenizer = tokenizer
        #self.sentence_db = replyvel.DB(Path(self.path, "texts", self.tokenizer.tag, "Sentence"), target_codes=self.target_codes)
        
    def get_normal_db(self):
        return JstatutreeDB(self.path, self.target_codes)

    @deprecate
    def tokenize_batch(self, overwrite=False, *args, **kwargs):
        with self.sentence_db.write_batch(*args, **kwargs) as wb:
            for e in self.iter_elements('Sentence'):
                if e.text is None:
                    wb.put(e.code, [])
                if overwrite or isinstance(e.text, str):
                    wb.put(e.code, self.tokenizer.tokenize(e.text))

    def export_corpus(self, unit, path=None, workers=None, batch_size=30):
        corpus_path = Path((path or self.path)/'corpus.txt')
        tag_path = Path((path or self.path)/'tags.txt')
        if tag_path.exists():
            print(tag_path, "already exists.")
            return

        splitted_dbs = [d for c, d in self.split_by_muni()]
        
        with open(corpus_path, 'w') as sf:
            with open(tag_path, 'w') as tf:
                for db_batch in [splitted_dbs[i:i+batch_size] for i in range(0, len(splitted_dbs), batch_size)]:
                    with concurrent.futures.ProcessPoolExecutor(workers) as executor:
                        futures = [executor.submit(SentenceGenerator(subdb, unit, True).sentence_list) for subdb in db_batch]
                        for future in concurrent.futures.as_completed(futures):
                            for tag, sent in future.result():
                                #print(sent)
                                sf.writelines(''.join(' '.join(sent).rstrip()+'\n'))
                                tf.writelines(tag+'\n')
    @deprecate
    def put_element(self, element):
        assert isinstance(element, Element)
        if element.etype == 'Sentence':
            self.sentence_db.put(element.code, self.tokenizer.tokenize(''.join(element.itertext())))
        for child in list(element):
            self.put_element(child)
        self.element_db[element.etype].put(element.code, element)
    
    def get_subdb(self, target_codes):
        return self.__class__(self.path, self.tokenizer, target_codes)

    # def get_jstatutree(self, code, default=None):
    #     code = str(ReikiCode(code))
    #     jstree = self.jstatutree_db.get(code, None)
    #     for e in jstree.iter('Sentence'):
    #         e.text = self.tokenizer.tokenize(e.text)
    #     if not jstree:
    #         return default
    #     #jstree._root = self.get_element(jstree._root, [])
    #     return jstree
    
    def _complete_element(self, elem):
        for e in elem.iter('Sentence'):
            if isinstance(e.text, str):
                e.text = self.tokenizer.tokenize(e.text)
                #print(e.text)
        return elem
    
    #def iter_elements(self, target_etype, include_tag=True):
    #    for elem in self.element_db[target_etype].iterator(include_key=False, include_value=True):
    #        yield self._complete_element(elem)

    def iter_jstatutree(self, include_tag=True, include_value=True, target_codes=None):
        for t in super().iter_jstatutree(include_tag, include_value, target_codes):
            t._root = self._complete_element(t._root)
            yield t


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
    
        
