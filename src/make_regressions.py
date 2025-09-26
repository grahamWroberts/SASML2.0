import argparse
import numpy as np
from sklearn.kernel_ridge import KernelRidge as KR
import xarray as XR
from sklearn.metrics import r2_score
import ArrayTuner
from scipy.stats.qmc import LatinHypercube
from sklearn.metrics import mean_absolute_error as MAE
import TreeHierarchy as TH
import json

def scale_range(val, bounds):
    return(val * (bounds[1]-bounds[0]) + bounds[0])

def scale_binary(val, bounds):
    return(bounds[0] if val < 0.5 else bounds[1])

def ident(x):
    return(x)

param_list = ["alpha", "gamma", "coef0"]
pbounds = {"alpha": (-3,3),
           "gamma": (-3,3),
           "coef0": (0,1)}
select_map = {"alpha": scale_range,
              "gamma": scale_range,
              "coef0": scale_binary}
wrapper_map = {"alpha": lambda x: 10**x,
               "gamma": lambda x: 100**x,
               "coef0": ident}

def map_rbf_KRR_sample(arr):
    return({param_list[i]: wrapper_map[param_list[i]](select_map[param_list[i]](arr[i], pbounds[param_list[i]])) for (i,p) in enumerate(arr)} | {"type": "KRR", "kernel": {"type": "rbf"}})

def InvertedKFold(X, y, n_splits=5):
    inds = np.arange(X.shape[0])
    np.random.shuffle(inds)
    xx = X.copy()
    cuts = np.linspace(0, X.shape[0], num=n_splits+1, dtype=int)
    for i in range(n_splits):
        train_inds = inds[cuts[i]:cuts[i+1]]
        val_inds = np.concatenate((inds[:cuts[i]],inds[cuts[i+1]:]))
        X_train = X[train_inds]
        y_train = y[train_inds]
        X_val = X[val_inds]
        y_val = y[val_inds]
        yield((X_train, y_train), (X_val, y_val))

def LHS_params(params, map_func, count=20):
    num_params = len(params)
    lh = LatinHypercube(num_params)
    outarr = [map_func(v) for v in lh.random(count)]
    return(outarr)

def create_krr(p):
    krr = KR(alpha = p["alpha"],
              gamma = p["gamma"],
              coef0 = p["coef0"],
              kernel = p["kernel"]["type"],
              degree = p["kernel"]["degree"] if p["kernel"]["type"] == "poly" else 0)
    return(krr)

def tune_reg(X, y, count = 5):
    params = LHS_params(param_list, map_rbf_KRR_sample, count=100)
    tuner = ArrayTuner.ArrayTuner(params)
    for p in tuner:
        err = np.zeros(count)
        for i, ((X_t, y_t), (X_v, y_v)) in enumerate(InvertedKFold(X, y, n_splits = count)):
            krr = create_krr(p)
            krr.fit(X_t, y_t)
            preds = krr.predict(X_v)
            err[i] = MAE(y_v, preds)
        tuner.report([p], np.mean(preds))
    return(tuner.best)

if __name__ == "__main__":
    indata = XR.load_dataset("data.nc")
    morphologies = np.unique(indata.labels.data)
    parameters = ['radius', 'length', 'thickness']
    train = np.log10(indata.SAS_curves.data)
    test = np.log10(indata.test_curves.data)
    regressors = {m : {} for m in morphologies}
    for m in morphologies:
        train_i = np.where(indata.labels.data == m)[0]
        test_i = np.where(indata.test_labels.data == m)[0]
        sel_curves = train[train_i,:]
        sel_test_curves = test[test_i,:]
        for p in parameters:
            sel_params = indata[p].data[train_i]
            sel_test_params = indata["test_%s"%(p)].data[test_i]
            if np.all(np.isfinite(sel_params)):
                #key = "%s_%s"%(p,m)
                best = tune_reg(sel_curves, sel_params)
                print(best)
                regressors[m][p]=create_krr(best[0]).fit(sel_curves, sel_params)
    encj = json.dumps(regressors, cls = TH.TreeEncoder, indent=3)
    outfile = open("regressors.json", 'w')
    outfile.write(encj)
    outfile.close()

