from threading import Lock, Thread
from queue import Queue

class QuickIterator(object):
    def __init__(self, kvsdb, batch_size=500):
        self.kvsdb = kvsdb
        self.mcodes = self.kvsdb.mcodes  #fix mcodes when this instance initialized
        self.jobq_lock = Lock()
        self.batch_size = batch_size
        self.init_queues()

    def init_queues(self):
        self.jobqueue = Queue()
        for mcode in self.mcodes:
            self.jobqueue.put(mcode)
        self.res_queue = Queue()
        
    def _get_available_db_from_jobq(self):
        self.jobq_lock.acquire()
        if self.jobqueue.empty():
            self.jobq_lock.release()
            return None
        first = None
        while True:
            mcode = self.jobqueue.get_nowait()
            #print(mcode)
            try:
                db = self.kvsdb.get_db(mcode, timeout=0)
                break
            except:
                self.jobqueue.put(mcode)
            if first == mcode:
                sleep(1)
            first = first or mcode
        self.jobq_lock.release()
        return db, mcode
    
    def _enqueue_items(self, include_key=True, include_value=True):
        while True:
            item = self._get_available_db_from_jobq()
            if item is None:
                break
            db, mcode = item
            for i, (item_key, value) in enumerate(db.iterator(include_key=True, include_value=True)):
                if i%self.batch_size == 0:
                    i > 0 and self.res_queue.put(batch)
                    batch = [None] * self.batch_size
                item = (self.kvsdb._decode_key(mcode, item_key), pickle.loads(value))
                batch[i%self.batch_size] = item
            self.res_queue.put([i for i in batch if i])
        self.res_queue.put(None)
                
    def items(self):
        self.init_queues()
        thread = Thread(target=self._enqueue_items)
        thread.start()
        while True:
            batch = self.res_queue.get()
            if batch is None:
                break
            for item in batch:
                yield item
        thread.join()
        raise StopIteration
    
    def _enqueue_keys(self):
        while True:
            item = self._get_available_db_from_jobq()
            if item is None:
                break
            db, mcode = item
            for i, item_key in enumerate(db.iterator(include_key=True, include_value=False)):
                if i%self.batch_size == 0:
                    i > 0 and self.res_queue.put(batch)
                    batch = [None] * self.batch_size
                item = (self.kvsdb._decode_key(mcode, item_key))
                batch[i%self.batch_size] = item
            self.res_queue.put([i for i in batch if i])
        self.res_queue.put(None)
                
    def keys(self):
        self.init_queues()
        thread = Thread(target=self._enqueue_keys)
        thread.start()
        while True:
            batch = self.res_queue.get()
            if batch is None:
                break
            for item in batch:
                yield item
        thread.join()
        raise StopIteration
        
    def _enqueue_values(self):
        while True:
            item = self._get_available_db_from_jobq()
            if item is None:
                break
            db, mcode = item
            for i, value in enumerate(db.iterator(include_key=False, include_value=True)):
                if i%self.batch_size == 0:
                    i > 0 and self.res_queue.put(batch)
                    batch = [None] * self.batch_size
                item = pickle.loads(value)
                batch[i%self.batch_size] = item
            self.res_queue.put([i for i in batch if i])
        self.res_queue.put(None)
                
    def values(self):
        self.init_queues()
        thread = Thread(target=self._enqueue_values)
        thread.start()
        while True:
            batch = self.res_queue.get()
            if batch is None:
                break
            for item in batch:
                yield item
        thread.join()
        raise StopIteration