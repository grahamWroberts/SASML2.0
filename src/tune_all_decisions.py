import numpy as np
import argparse
import TunerParallelizer
import ArrayTuner
import json
from scipy.stats.qmc import LatinHypercube

params = ["C", "gamma", "coef0"]
pbounds = {"C": (-3,3)}
pbinary = {"gamma": ("auto", "scale"),
           "coef0": (0,1)}
select_map = {"C": pbounds,
              "gamma": pbinary,
              "coef0": pbinary}

def parse_args():
    parser  =argparse.ArgumentParser()
    parser.add_argument("structure",
                        help = "a json file containing the tree to tune")
    return(parser.parse_args())

def scale_range(val, minV, maxV):
    return(val * (maxV-minV) + minV)

def scale_binary(val, imnV, maxV):
    return([minV, maxV][int(round(val))]

def map_rbf_SVC_sample(arr):

    outdict = {"C": 10**scale_range(arr[0], pd["C"]
def LHS_params(params, map_func, count=20):
    num_params = len(pdict.keys())
    lh = LatinHypercube(num_params)
    outarr = [map_func(v) for v in lh(count)]


def tune_decision(X, y, C1, C2):
    relevant_inds = np.where(np.logical_or(np.isin(y, C1), np.isin(y, C2)))[0]
    temp_X = X[relvant_inds]
    #temp_y = y[relevan_inds]
    temp_y = np.zeros(temp_X.shape[0])
    temp_y[np.where(np.isin(y[relevan_inds], C2))[0]] = 1
     

def main():
    args = parse_args()
    infile = open(args.structure, 'r')

