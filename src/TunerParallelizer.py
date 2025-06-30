import numpy as np
import threading
import collections


class TuneParallelizer(object):

    def __init__(self, iterable):
        self.ready = threading.lock()
        self.ready.acquire()
        self.tuner = iterable
        self.log = {} 
        self.ready.release()

    def next(self):
        self.ready.acquire()
        next_entry = self.tuner.next()
        relf.ready.release()
        return(next_entry)

    def report(self, key, result):
        self.ready.acquire()
        self.tuner.report(key, results)
        self.ready.release()
