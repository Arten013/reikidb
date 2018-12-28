from .utils import db_unit
from .utils import qiterator

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
from .utils.logger import get_logger
from typing import Sequence

"""
def db_freezer(*dbs):
    def _db_freezer(func):
        def wrapper(self, *args, **kwargs):
            for db in dbs:
                db.
"""


class DB(object):
    def __init__(self, basepath, target_codes='ALL',* , auto_release_interval=3, db_acquire_timeout=5, value_encoder=None, value_decoder=None):
        self.basepath = Path(basepath)
        self.target_codes = target_codes
        self.target_mcode_ptns = self._get_mcode_ptns(target_codes)
        self._auto_release_interval = auto_release_interval
        self._db_acquire_timeout = db_acquire_timeout
        self._db_cache = dict()
        self.value_encoder = value_encoder or pickle.dumps
        self.value_decoder = value_decoder or pickle.loads

    def split_unit(self):
        return [self.get_subdb([mcode]) for mcode in self.mcodes]

    def get_subdb(self, target_codes):
        subdb_mcodes = [code for code in target_codes if self._code_ptn_match(code)]
        if len(subdb_mcodes) == 0:
            raise ValueError('Invalid code: ' + ', '.join(target_codes))
        return self.__class__(self.basepath, subdb_mcodes, auto_release_interval=self._auto_release_interval, db_acquire_timeout=self._db_acquire_timeout)

    def suspend_releaser(self):
        self._auto_release_interval = db_unit.INF_RELEASE_TIME
        for v in self._db_cache.values():
            v.suspend_releaser()

    @staticmethod
    def _get_mcode_ptns(codes):
        if codes == 'ALL':
            return 'ALL'
        ptns = []
        for code in codes:
            if re.match('\d{2}$', code):
                ptns.append(re.compile(code + '\d{4}'))
            elif re.match('\d{6}$', code):
                ptns.append(re.compile(code))
        return ptns

    def __len__(self):
        return sum(1 for _ in self.iterator(include_key=False, include_value=False))

    def _db_unit_factory(self, mcode):
        return db_unit.DBUnit(self.get_db_path(mcode), auto_release_interval=self._auto_release_interval,
                              default_timeout=self._db_acquire_timeout)

    def remove_files(self):
        shutil.rmtree(self.basepath)

    def _code_ptn_match(self, code, ptns=None):
        ptns = ptns or self.target_mcode_ptns
        if ptns == 'ALL':
            return True
        for ptn in ptns:
            if ptn.match(code):
                return True
        return False

    @property
    def mcodes(self):
        if self.target_mcode_ptns == 'ALL':
            return self._exist_mcodes()
        else:
            return [mcode for mcode in self._exist_mcodes() if self._code_ptn_match(mcode)]

    _mcode_ptn = re.compile('\d{6}$')

    def _exist_mcodes(self):
        if not self.basepath.exists():
            return []
        return [x.name for x in self.basepath.iterdir() if x.is_dir() and re.match(self.__class__._mcode_ptn, x.name)]

    def get_db_path(self, mcode):
        return self.basepath / Path(mcode)

    def get_db(self, mcode, timeout=None):
        if mcode not in self._db_cache:
            dbpath = self.get_db_path(mcode)
            if not dbpath.exists():
                os.makedirs(str(dbpath))
            self._db_cache[mcode] = self._db_unit_factory(mcode)
        self._db_cache[mcode].acquire(timeout=timeout)
        return self._db_cache[mcode]

    def get(self, key, default=None):
        # print('replyvel get:', key)
        mcode, item_key = self._encode_key(key)
        if not self.get_db_path(mcode).exists():
            return default
        db = self.get_db(mcode)
        value = db.get(item_key, default=None)
        if value is None:
            # print('replyvel get (return default):', key)
            return default
        ret = self._decode_value(value)
        return ret

    def put(self, key, value):
        mcode, item_key = self._encode_key(key)
        db = self.get_db(mcode)
        db.put(item_key, self._encode_value(value))

    def delete(self, key):
        mcode, item_key = self._encode_key(key)
        db = self.get_db(mcode)
        db.delete(item_key)
        for i in db.iterator(include_key=True, include_value=False):
            break
        else:
            shutil.rmtree(self.get_db_path(mcode))

    @classmethod
    def _encode_key(cls, key):
        parts = Path(key).parts
        #print(parts)
        return parts[1], os.path.join(*parts[2:]).encode()

    @classmethod
    def _decode_key(cls, mcode, item_key):
        return os.path.join(mcode[:2], mcode, item_key.decode())

    def _encode_value(self, value):
        return self.value_encoder(value)

    def _decode_value(self, value):
        return self.value_decoder(value)

    def is_empty(self):
        return len(self.mcodes) == 0

    def _get_item_unpacker(self, include_key, include_value):
        if include_key and include_value:
            return lambda mcode, kv: (self._decode_key(mcode, kv[0]), self._decode_value(kv[1]))
        elif include_key:
            return lambda mcode, key: self._decode_key(mcode, key)
        elif include_value:
            return lambda mcode, value: self._decode_value(value)
        else:
            return lambda mcode, value: None

    def iterator(self, include_key: bool=True, include_value: bool=True, target_codes: Sequence[str] = None, prefix: str = None):
        logger = get_logger('_replyvel.DB.iterator')
        task_queue = Queue()
        target_mcode_ptns = self._get_mcode_ptns(target_codes) if target_codes else None
        for mcode in sorted(
                [
                    mc for mc in self.mcodes
                    if self._code_ptn_match(mc, ptns=target_mcode_ptns)
                    and (prefix is None or mc in prefix)
                ]
                , key=lambda x: int(x)):
            task_queue.put(mcode)
        unpacker = self._get_item_unpacker(include_key, include_value)
        while not task_queue.empty():
            db = None
            mcode = None
            while not db:
                mcode = task_queue.get()
                try:
                    db = self.get_db(mcode)
                except:
                    task_queue.put(mcode)
            logger.debug('iterate from: ' +mcode)
            processed_prefix = self._encode_key(prefix)[1] if prefix is not None else None
            logger.info('prefix: %s(%s)', prefix or 'None', processed_prefix or 'None')
            for item in db.iterator(include_key=include_key, include_value=include_value, prefix=processed_prefix):
                try:
                    yield unpacker(mcode, item)
                except Exception as e:
                    print(item)
                    raise

    def items(self):
        yield from self.iterator(include_key=True, include_value=True)

    def keys(self):
        yield from self.iterator(include_key=True, include_value=False)

    def values(self):
        yield from self.iterator(include_key=False, include_value=True)

    def qitems(self):
        yield from qiterator.QuickIterator(self, mcodes=None, batch_size=500).items()

    def qkeys(self):
        yield from qiterator.QuickIterator(self, mcodes=None, batch_size=500).keys()

    def qvalues(self):
        yield from qiterator.QuickIterator(self, mcodes=None, batch_size=500).values()

    def write_batch_mapping(self, mapping, *args, **kwargs):
        with BatchWriter(self, *args, **kwargs) as wv:
            for k, v in mapping.items():
                wv.put(k, v)

    def write_batch(self, *args, **kwargs):
        return BatchWriter(self, *args, **kwargs)


class BatchWriter(object):
    def __init__(self, replyvel_db, *wb_args, **wb_kwargs):
        self.db = replyvel_db
        self.wb_args = wb_args
        self.wb_kwargs = wb_kwargs
        self.wbdict = dict()
        self.dbdict = dict()

    def put(self, key, value):
        mcode, item_key = self.db._encode_key(key)
        if mcode not in self.wbdict:
            db = self.db.get_db(mcode)
            self.wbdict[mcode] = db.write_batch(*self.wb_args, **self.wb_kwargs)
            self.dbdict[mcode] = db
        # print(mcode, item_key, value)
        self.wbdict[mcode].put(item_key, self.db._encode_value(value))

    def delete(self, key):
        mcode, item_key = self.db._encode_key(key)
        if mcode not in self.wbdict.keys():
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
                shutil.rmtree(self.db.get_db_path(mcode))
        self.dbdict = {}
