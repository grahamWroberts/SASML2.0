import numpy as np
import sys

def gen_factory(arr):
    i=0
    lim = len(arr)
    while i<lim:
        yield(arr[i])
        i += 1

class ArrayTuner(object):

    def __init__(self, arr):
       self._arr = arr
       self.result = np.zeros(len(arr))
       self.iter = gen_factory(arr)
       self.reference = arr
       self.best_perf = 0
       self.best = None
       return

    def report(self,vec, res):
        if len(vec) > 1:
           matches = np.ones(self.result.shape[0])
           for i in range(len(vec)):
               matches = np.logical_and(matches, np.equal(self.reference[:,i], vec[i]))
        else:
            matches = np.equal(self.reference, vec)
        if np.sum(matches)>0:
            ind = np.where(matches)[0][0]
            self.result[ind] = res
            if res > self.best_perf:
                self.best_perf = res
                self.best = vec
        else:
            print("ERROR UNABLE TO LOCATE RECORD!!!", sys.stderr)
        return

    def reset(self):
        self.iter = gen_factory(self.reference)

    def next(self):
        return(next(self.iter))

    def __next__(self):
        return(self.next())

    def __iter__(self):
        return(self)

    

        


