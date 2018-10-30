import os
import re
from pathlib import Path
import plyvel
from threading import Lock, Condition, Thread
from queue import Queue
import shutil
import pickle
from datetime import datetime, timedelta
from time import sleep
import traceback

def print_deco(func):
    def wrapper(self, *args, **kwargs):
        func.__name__ != 'wait' and print(func.__name__)
        return func(self, *args, **kwargs)
    return wrapper

class PrintCondition(object):
    def __init__(self, *args, **kwargs):
        self.lock = Condition(*args, **kwargs)
    
    @print_deco
    def acquire(self, *args, **kwargs):
        return self.lock.acquire(*args, **kwargs)
        
    @print_deco
    def release(self, *args, **kwargs):
        return self.lock.release(*args, **kwargs)

    @print_deco
    def wait(self, *args, **kwargs):
        return self.lock.wait(*args, **kwargs)
        
    @print_deco
    def notify(self, *args, **kwargs):
        return self.lock.notify(*args, **kwargs)

def lock_deco(func):
    def wrapper(self, *args, **kwargs):
        _from_outside = kwargs.get('_from_outside', True)
        if _from_outside:
            self.lock.acquire()
        ret = func(self, *args, **kwargs)
        if _from_outside:
            self.lock.release()
        return ret
    return wrapper

def load_leveldb_deco(release_interval=None, timeout=None):
    def _load_leveldb_deco(func):
        def wrapper(self, *args, **kwargs):
            if not self.acquire(release_interval=release_interval, timeout=timeout):
                raise LevelDBAcquisitionFailure('Failed to acquire db "{}"'.format(self.path))
            return func(self, *args, **kwargs)
        return wrapper
    return _load_leveldb_deco

INF_RELEASE_TIME = -1

class WriteBatch(object):
    def __init__(self, udb, *args, **kwargs):
        self.udb = udb
        self.wb = udb.db.write_batch(*args, **kwargs)
        self._restart_releaser = False
        
    def close(self):
        self.wb.close()
        if self._restart_releaser:
            self.udb.release()
    
    def __enter__(self):
        if not self.udb.releaser_is_suspended():
            self.udb.suspend_releaser()
            self._restart_releaser = True
        return self.wb.__enter__()
    
    def __exit__(self, *args, **kwargs):
        self.wb.__exit__(*args, **kwargs)
        if self._restart_releaser:
            self.udb.release()
        
    def __getattr__(self, attr):
        return getattr(self.wb, attr)
    
class Iterator(object):
    def __init__(self, udb, *args, **kwargs):
        self.udb = udb
        self.args = args
        self.kwargs = kwargs
        self._restart_releaser = False
        
    def __iter__(self):
        if not self.udb.releaser_is_suspended():
            self.udb.suspend_releaser()
            self._restart_releaser = True
        yield from self.udb.db.iterator(*self.args, **self.kwargs)
        if self._restart_releaser:
            self.udb.release()

class DBUnit(object):
    __slots__ = [
        "default_timeout",
        'is_closed',
        "lock",
        "path",
        "_auto_release_interval",
        "_auto_releaser",
        "_db",
        "_releaser_cond",
        "_release_time",
        "_retry_interval"
    ]
    
    def __init__(self, path, auto_release_interval=5, retry_interval=0.1, default_timeout=1):
        self.path = Path(path)
        self._auto_release_interval = auto_release_interval
        self._release_time = INF_RELEASE_TIME
        self._db = None
        self._retry_interval = retry_interval
        self.lock = Condition()
        #self._releaser_cond = Condition(lock=self.lock)
        self.default_timeout = default_timeout
        self.is_closed = False
        self._auto_releaser = Thread(target=self._auto_release)
        self._auto_releaser.start()
    
    @lock_deco
    def close(self, _from_outside=True):
        self.is_closed = True
        if self._auto_releaser:
            self.release()
            self.lock.release()
            self._auto_releaser.join()
            self.lock.acquire()

    def releaser_is_suspended(self):
        return self._release_time == INF_RELEASE_TIME
            
    @lock_deco
    def suspend_releaser(self, acquire=True, timeout=None, _from_outside=True):
        if self.is_acquired(_from_outside=False):
            self._set_release_time(INF_RELEASE_TIME, _from_outside=False)
        elif acquire:
            self.acquire(release_interval=INF_RELEASE_TIME, timeout=timeout, _from_outside=False)

    @lock_deco
    def restart_releaser(self, release_interval=None, acquire=True, timeout=None, _from_outside=True):
        if self.is_acquired(_from_outside=False):
            self._set_release_time(release_time=release_interval or self._auto_release_interval, _from_outside=False)
        elif acquire:
            self.acquire(release_interval=release_interval, timeout=timeout, _from_outside=False)
            
    @lock_deco
    def release(self, _from_outside=True):
        if self.is_acquired(_from_outside=False):
            self._set_release_time(datetime.now(), _from_outside=False)

    @lock_deco
    def _set_release_time(self, release_time, _from_outside=True):
        if release_time == INF_RELEASE_TIME:
            print('released reservation: Unlimited')
            self._release_time = INF_RELEASE_TIME
            return
        if isinstance(release_time, (int, float)):
            release_time = max(timedelta(seconds=0), timedelta(seconds=release_time)) + datetime.now()
        if isinstance(release_time, timedelta):
            release_time = release_time + datetime.now()
        elif self._release_time == INF_RELEASE_TIME \
            or (release_time > self._release_time and (release_time - self._release_time).seconds > 0.1)\
            or (release_time < self._release_time and (self._release_time - release_time).seconds > 0.1):
            print('released reservation:  ', release_time)
            self._release_time = release_time
            self.lock.notify()
        else:
            pass
            #print('No reservation')

    def _wait_predicate(self):
        if self._db is None:
            return True
        elif self._release_time is INF_RELEASE_TIME:
            return True
        elif self._release_time > datetime.now():
            return True
        return False
            
    def _auto_release(self):
        print('auto releaser launched')
        self.lock.acquire()
        while not self.is_closed:
            rt = self._release_time
            rt != INF_RELEASE_TIME and print('next release_time:', rt)
            while self._wait_predicate():
                if self._release_time != rt:
                    #print('changed from', rt, 'to', self._release_time)
                    rt = self._release_time
                self.lock.wait(timeout=0.1)
            self._db.close()
            del self._db
            self._db = None
            self._release_time = INF_RELEASE_TIME
            print('*released')
        self.lock.release()
        print('auto releaser finished')

    @property
    def db(self):
        return self._db or {}
    
    @lock_deco
    def acquire(self, release_interval=None, timeout=None, _from_outside=True):
        timeout = max(timeout or self.default_timeout, 0)
        time_limit = datetime.now() + timedelta(seconds=timeout)
        release_interval = release_interval or self._auto_release_interval
        #print('release interval:', release_interval)
        first_flag=True
        while first_flag or time_limit > datetime.now():
            first_flag=False
            # exm = None
            try:
                if not self.is_acquired(_from_outside=False):
                    print('try to open database', self.path)
                    self._db = plyvel.DB(str(self.path), create_if_missing=True)
                    print('successfully opened db', self.path)
                self._set_release_time(release_interval, _from_outside=False)
                return True
            except plyvel.Error as e:
                # exm = traceback.format_exc()
                sleep(self._retry_interval)
                # timeout >= 0 and print('retry')
                continue
        return False

    @lock_deco
    def is_acquired(self, _from_outside=True):
        return self._db is not None

    @load_leveldb_deco()
    def get(self, key, default=None):
        return self.db.get(key, default)

    @load_leveldb_deco()
    def put(self, key, value):
        return self.db.put(key, value)

    @load_leveldb_deco()
    def delete(self, key):
        return self.db.delete(key)

    @load_leveldb_deco(release_interval=INF_RELEASE_TIME)
    def iterator(self, include_key, include_value):
        return Iterator(self, include_key=include_key, include_value=include_value)

    @load_leveldb_deco(release_interval=INF_RELEASE_TIME)
    def write_batch(self, *args, **kwargs):
        return WriteBatch(self, *args, **kwargs)

    @load_leveldb_deco(release_interval=INF_RELEASE_TIME)
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type:
            raise
        self._set_release_time(self._auto_release_interval)


class LevelDBAcquisitionFailure(Exception):
    pass

