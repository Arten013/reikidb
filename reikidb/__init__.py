from . import db_unit
from . import qiterator
from . import replyvel
from . import dataset
from . import tokenizer

import traceback
import os
import re
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import plyvel
from threading import Lock, Thread
from queue import Queue
import shutil
import pickle


mcode_ptn = re.compile('\d{6}$')
def iter_mcodes(path):
    path = Path(path)
    if mcode_ptn.match(path.name):
        yield path.name
    else:
        for child_path in Path(path).iterdir():
            yield from iter_mcodes(child_path)

class ReplyvelDict(object):
    def __init__(self, path, target_codes='ALL', create_if_missing=True):
        self.path = path
        self.db = replyvel.DB(self.path, target_codes=target_codes)
        if create_if_missing:
            os.makedirs(self.path, exist_ok=True)
        
    def to_dict(self):
        return {k:v for k, v in self.items()}

    def __setitem__(self, key, val):
        self.db.put(key, val)

    def __getitem__(self, key):
        return self.db.get(key)

    def __delitem__(self, key):
        self.db.delete(key)

    def get(self, key, default=None):
        return self.db.get(key, default=default)
        
    def __len__(self):
        return sum(1 for _ in self.keys())
    
    def is_prefixed_db(self):
        return False

    def is_empty(self):
        return self.db.is_empty()
    
    def write_batch_mapping(self, mapping, *args, **kwargs):
        with BatchDictWriter(self.db) as wv:
            for k, v in mapping.items():
                wv[k] = v
            
    def write_batch(self, *args, **kwargs): 
        return BatchDictWriter(self.db, *args, **kwargs)
    
    def items(self):
        return self.db.items()

    def keys(self):
        return self.db.keys()

    def values(self):
        return self.db.values()

    def qitems(self):
        return self.db.qitems()

    def qkeys(self):
        return self.db.qkeys()

    def qvalues(self):
        return self.db.qvalues()

class BatchDictWriter(replyvel.BatchWriter):
    __setitem__ = replyvel.BatchWriter.put
    __delitem__ = replyvel.BatchWriter.delete
    
