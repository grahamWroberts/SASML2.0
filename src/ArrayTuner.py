import numpy as np
import sys

def gen_factory(arr):
    i=0
    lim = arr.shape[0]
    while i<lim:
        yield(arr[i])
        i += 1

class ArrayTuner(object):

    def __init__(self, arr):
       self._arr = arr
       self.result = np.zeros(arr.shape[0])
       self.iter = gen_factory(arr)
       return(self)

    def report(vec, res):
        matches = np.ones(self.results.shape[0])
        for i in range(vec.shape[0]):
            matches = np.logical_and(ind, np.equal(self.arr[:,i], vec[i]))
        if np.sum(matches)>0:
            ind = np.where(matches)[0][1]
            self.result[ind] = res
        else:
            print("ERROR UNABLE TO LOCATE RECORD!!!", sys.stderr)
        return()

    

        


