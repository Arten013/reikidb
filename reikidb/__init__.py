from . import db_unit

import traceback
import os
import re
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import plyvel
from threading import Lock
from queue import Queue
import shutil
import pickle

class ReikiKVSDictDB(object):
    def __init__(self, basepath, thread_num=5, auto_release_interval=60, db_acquire_timeout=5):
        self.basepath = Path(basepath)
        self.thread_num = thread_num
        self.workers = ThreadPoolExecutor(max_workers=self.thread_num)
        self._auto_release_interval = auto_release
        self._db_acquire_timeout = db_acquire_timeout
        self._db_cache = dict()

    def _dbunit_factory(self, mcode):
        return db_unit.DBUnit(self.get_db_path(mcode), self._auto_release_interval, self._db_acquire_timeout)
    
    def remove_files(self):
        shutil.rmtree(self.basepath)
    
    mcode_ptn = re.compile('\d{6}')
    @property
    def mcodes(self):
        if not self.basepath.exists():
            return []
        return [x.name for x in self.basepath.iterdir() if x.is_dir() and re.match(self.__class__.mcode_ptn, x.name)]
        
    def get_db_path(self, mcode):
        return self.basepath / Path(mcode)
        
    def get_db(self, mcode):
        if mcode not in self._db_cache:
            dbpath = self.get_db_path(mcode)
            if not dbpath.exists():
                os.makedirs(str(dbpath))
            self._db_cache[mcode] = self._db_unit_factory(mcode)
        return self._db_cache[mcode]
        
    def __getitem__(self, key):
        ret = self.get(key)
        if ret is None:
            raise KeyError(key, 'is not registered in the db', self.basepath)
        return ret

    def get(self, key, default=None):
        mcode, item_key = self._encode_key(key)
        if not self.get_db_path(mcode).exists():
            return default
        db = self.get_db(mcode)
        ret =  pickle.loads(db.get(item_key, default=default))
        return ret
    
    def put(self, key, value):
        mcode, item_key = self._encode_key(key)
        db = self.get_db(mcode)
        db.put(item_key, pickle.dumps(value))
    
    def delete(self, key):
        mcode, item_key = self._encode_key(key)
        db = self.get_db(mcode)
        db.delete(item_key)
        for i in db.iterator(include_key=True, include_value=False):
            break
        else:
            shutli.rmtree(self.get_db_path(mcode))
    
    @staticmethod
    def _encode_key(key):
        parts = Path(key).parts
        return parts[1], os.path.join(*parts[2:]).encode()
    
    @staticmethod
    def _decode_key(mcode, item_key):
        return os.path.join(mcode[:2], mcode, item_key.decode())
    
    def is_empty(self):
        return len(self.mcodes) == 0
    
    def items(self):
        for mcode in sorted(self.mcodes, key=lambda x:int(x)):
            db = self.get_db(mcode)
            for item_key, value in db.iterator(include_key=True, include_value=True):
                yield self._decode_key(mcode, item_key), pickle.loads(value)

    def keys(self):
        for mcode in sorted(self.mcodes, key=lambda x:int(x)):
            db = self.get_db(mcode)
            for item_key in db.iterator(include_key=True, include_value=False):
                yield self._decode_key(mcode, item_key)

    def values(self):
        for mcode in sorted(self.mcodes, key=lambda x:int(x)):
            db = self.get_db(mcode)
            for value in db.iterator(include_key=False, include_value=True):
                yield pickle.loads(value)
    
    def _qiter_base(self, enqueue_func):
        queue = Queue()
        enqueue_func.queue = queue
        mcodes = self.mcodes
        for mcode in sorted(mcodes, key=lambda x:int(x)):
            self.workers.submit(enqueue_func, mcode)
        running_count = len(mcodes)
        while running_count > 0:
            item = queue.get()
            if item is None:
                running_count -= 1
            else:
                yield item
    
    def qitems(self):
        def enqueue_func(mcode):
                queue = enqueue_func.queue
                db = self.get_db(mcode)
                for item_key, value in db.iterator(include_key=True, include_value=True):
                    queue.put((self._decode_key(mcode, item_key), pickle.loads(value)))
                queue.put(None)
        for k, v in self._qiter_base(enqueue_func):
            yield k, v

    def qkeys(self):
        def enqueue_func(mcode):
                queue = enqueue_func.queue
                db = self.get_db(mcode)
                for item_key in db.iterator(include_key=True, include_value=False):
                    queue.put(self._decode_key(mcode, item_key))
                queue.put(None)
        yield from self._qiter_base(enqueue_func)

    def qvalues(self):
        def enqueue_func(mcode):
                queue = enqueue_func.queue
                db = self.get_db(mcode)
                for value in db.iterator(include_key=False, include_value=True):
                    queue.put(pickle.loads(value))
                queue.put(None)
        yield from self._qiter_base(enqueue_func)

class BatchReikiWriter(object):
    def __init__(self, reikikvsdb, *wb_args, **wb_kwargs):
        self.db = reikikvsdb
        self.wb_args = wb_args
        self.wb_kwargs = wb_kwargs
        self.wbdict = dict()
        self.dbdict = dict()
    
    def __setitem__(self, key, value):
        mcode, item_key = self.db._encode_key(key)
        if mcode not in self.wbdict:
            db = self.db.get_db(mcode)
            self.wbdict[mcode] = db.write_batch(*self.wb_args, **self.wb_kwargs)
            self.dbdict[mcode] = db
        #print(mcode, item_key, value)
        self.wbdict[mcode].put(item_key, pickle.dumps(value))

    def __delitem__(self, key):
        mcode, item_key = self.db._encode_key(key)
        if mcode not in self.wbdict():
            db = self.db.get_db(mcode)
            self.wbdict[mcode] = db.write_batch(*self.wb_args, **self.wb_kwargs)
            self.dbdict[mcode] = db
        self.wbdict[mcode].delete(item_key)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is not None:
            return False
        self.write()
        return True
    
    
    def write(self):
        for wb in self.wbdict.values():
            wb.write()
        self.wbdict = {}
        for mcode, db in self.dbdict.items():
            for i in db.iterator(include_key=True, include_value=False):
                db.close()
                break
            else:
                db.close()
                shutli.rmtree(self.db.get_db_path(mcode))
        self.dbdict = {}
    

class ReikiKVSDict(object):
    ENCODING = "utf8"

    def __init__(self, path, thread_num=None, create_if_missing=True):
        self.path = path
        self.thread_num = thread_num or 5
        self.db = ReikiKVSDictDB(self.path, self.thread_num)

        if create_if_missing:
            os.makedirs(self.path, exist_ok=True)
    
    @property
    def path(self):
        if "_path" not in self.__dict__:
            self._path = None
        return self._path

    @path.setter
    def path(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if os.path.splitext(path)[1] == "":
            path += ".ldb"
        self._path = path
        
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
        l = 0
        for _ in self.qkeys():
            l += 1
        return l
    
    def is_prefixed_db(self):
        return False

    def is_empty(self):
        return self.db.is_empty()
    
    def write_batch_mapping(self, mapping, *args, **kwargs):
        with BatchReikiWriter(self.db) as wv:
            for k, v in mapping.items():
                wv[k] = v
            
    def write_batch(self, *args, **kwargs): 
        return BatchReikiWriter(self.db, *args, **kwargs)
    
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




import re, pickle
class Lawcodes(object):
    def __init__(self, path):
        self.basepath = Path(path)
        self.lcdict = {}
        self.changed_list = []
    
    mcode_ptn = re.compile('\d{6}')
    @property
    def mcodes(self):
        yield from self.changed_list
        if not self.basepath.exists():
            return []
        yield from (x.stem for x in self.basepath.iterdir() if not x.is_dir() and re.match(self.__class__.mcode_ptn, x.stem) and x.stem not in self.changed_list)

    def get_path(self, mcode=None, pickle_file=False):
        if mcode:
            mcode = str(mcode)
            if pickle_file:
                return self.basepath/Path(mcode+'.pkl')
            else:
                return self.basepath/Path(mcode)
        else:
            return self.basepath
    
    def __contains__(self, item):
        parts = Path(item).parts
        if len(parts) < 3:
            return False
        pcode, mcode, fcode = parts[:3]
        self.lcdict[pcode] = self.lcdict.get(pcode, dict())
        self.lcdict[pcode][mcode] = self.lcdict[pcode].get(mcode, self._load_sub(mcode)) 
        return fcode in self.lcdict[pcode][mcode] 
    
    def _load_sub(self, mcode):      
        path = self.get_path(mcode, True)
        if not path.exists() or not os.path.getsize(path):
            return []
        with open(path, 'rb') as f:
            return pickle.load(f)

    def __iter__(self):
        for mcode in self.mcodes:
            pcode = mcode[:2]
            self.lcdict[pcode] = self.lcdict.get(pcode, dict())
            d = self.lcdict[pcode].get(mcode, self._load_sub(mcode))
            yield from (os.path.join(pcode, mcode, fcode) for fcode in d)

    def __len__(self):
        return sum(1 for v in self)

    def append(self, key):
        if key in self:
            return
        parts = Path(key).parts
        if len(parts) != 3:
            return
        pcode, mcode, fcode = parts
        self.lcdict[pcode] = self.lcdict.get(pcode, dict())
        self.lcdict[pcode][mcode] =self.lcdict[pcode].get(mcode, self._load_sub(mcode)) + [fcode]
        if mcode not in self.changed_list:
            self.changed_list.append(mcode)
    
    def __getitem__(self, key):
        key = str(key)
        if re.match('\d{2}$', key):
            return self.lcdict(key)
        if re.match('\d{6}$', key):
            try:
                return self.lcdict[key[:2]][key]
            except KeyError:
                raise KeyError(key)
        if re.match('(\d{2}/)\d{6}$', key):
            mcode, fcode = Path(key).parts
            try:
                return self.lcdict[mcode][fcode]
            except KeyError:
                raise KeyError(key)
        raise KeyError(key)
    
    def write(self):
        os.makedirs(self.basepath, exist_ok=True)
        for mcode in self.changed_list:
            with open(self.get_path(mcode, True), 'wb') as f:
                pickle.dump(self[mcode], f)
                
