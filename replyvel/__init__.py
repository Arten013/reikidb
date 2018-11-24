from ._replyvel import DB
from . import _replyvel as replyvel
from . import dataset
from . import tokenizer
from . import model_core
from . import ml_models
from . import policy_match

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


"""
# fasttext training
import subprocess
import traceback
vocab_size = 8000
spm_model_tag = str(vocab_size)
db_path = Path('./../tokai_db')
fasttext_path = Path('./../tokai_db/fasttext/')
reikiset_path = Path('./../reikiset/')
spm = SPTokenizer(Path(db_path, 'spm',  spm_model_tag))
tdb = replyvel.dataset.TokenizedJstatutreeDB(db_path, tokenizer=spm)


def export_sentence_corpus(source_db, target_path, overwrite=False):
    if not Path(fasttext_path, 'corpus.text').exists():
        os.makedirs(fasttext_path, exist_ok=True)
        with open(Path(fasttext_path, 'corpus.text'), 'w') as f:
            f.writelines(' '.join(words).rstrip()+'\n' 
                         for words in tdb.sentence_db.iterator(include_key=False) if len(words)
                        )
    
    
try:
    if not Path(fasttext_path, 'model.ft').exists():
        if not Path(fasttext_path, 'corpus.text').exists():
            os.makedirs(fasttext_path, exist_ok=True)
            with open(Path(fasttext_path, 'corpus.text'), 'w') as f:
                f.writelines(' '.join(words).rstrip()+'\n' 
                             for words in tdb.sentence_db.iterator(include_key=False) if len(words)
                            )
            send_slack('Corpus preparation finished')
        cp = subprocess.run('/home/jovyan/fastText-0.1.0/fasttext skipgram -input ./corpus.txt -output ./model.ft' ,shell=True, cwd=fasttext_path)
        cp.check_returncode()
    send_slack('Model fitting finished')
except:
    send_slack(traceback.format_exc())
    send_slack(cp.stdout)
    send_slack(cp.stderr)
"""




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
    
