import numpy as np
import argparse
import TunerParallelizer
import ArrayTuner
import json
from scipy.stats.qmc import LatinHypercube
import TreeHierarchy as TH
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

def scale_range(val, bounds):
    return(val * (bounds[1]-bounds[0]) + bounds[0])

def scale_binary(val, bounds):
    return(bounds[0] if val < 0.5 else bounds[1])

def ident(x):
    return(x)

param_list = ["C", "gamma", "coef0"]
pbounds = {"C": (-3,3),
           "gamma": ("auto", "scale"),
           "coef0": (0,1)}
select_map = {"C": scale_range,
              "gamma": scale_binary,
              "coef0": scale_binary}
wrapper_map = {"C": lambda x: 10**x,
               "gamma": ident,
               "coef0": ident}


def parse_args():
    parser  =argparse.ArgumentParser()
    parser.add_argument("structure",
                        help = "a json file containing the tree to tune")
    parser.add_argument("--X", help = "a file containing data features", default = "X.csv")
    parser.add_argument("--y", help = "a file containing data labels", default = "y.csv")
    parser.add_argument("--test_X", help = "a file containing data features of the test set", default = "X_test.csv")
    parser.add_argument("--test_y", help = "a file containing data labels of the test set", default = "y_test.csv")
    parser.add_argument("--output", help = " a file to save tuned tree to", default = "tuned_tree.json")
    return(parser.parse_args())

def map_rbf_SVC_sample(arr):
    return({param_list[i]: wrapper_map[param_list[i]](select_map[param_list[i]](arr[i], pbounds[param_list[i]])) for (i,p) in enumerate(arr)} | {"type": "SVC", "kernel": {"type": "rbf"}})

def LHS_params(params, map_func, count=20):
    num_params = len(params)
    lh = LatinHypercube(num_params)
    outarr = [map_func(v) for v in lh.random(count)]
    return(outarr)

def load_all_data(args):
    X = np.log10(np.loadtxt(args.X, delimiter=','))
    y = np.loadtxt(args.y, delimiter=',', dtype = str)
    test_X = np.log10(np.loadtxt(args.test_X, delimiter=','))
    test_y = np.loadtxt(args.test_y, delimiter=',', dtype = str)
    return(X, y, test_X, test_y)

def select_data(X, y, C1, C2):
    relevant_inds = np.where(np.logical_or(np.isin(y, C1), np.isin(y, C2)))[0]
    temp_X = X[relevant_inds]
    #temp_y = y[relevan_inds]
    temp_y = np.zeros(temp_X.shape[0])
    temp_y[np.where(np.isin(y[relevant_inds], C2))[0]] = 1
    return(temp_X, temp_y)

def parse_json(fn):
    infile = open(fn, "r")
    indat = infile.read()
    j_string = json.loads(indat)
    return(j_string)

def make_tree(fn):
    jdict = parse_json(fn)
    tree = TH.TreeHierarchy()
    #tree.from_json(jdict)
    tree.structure_from_json(jdict)
    return(tree)

def inverse_KFold(X, y, classifier, n_splits=5):
    skf = StratifiedKFold(n_splits = n_splits, shuffle=True)
    sum_perf = 0
    for val_inds, train_inds in skf.split(X,y):
        classifier.fit(X[train_inds], y[train_inds])
        sum_perf += accuracy_score(y[val_inds], classifier.predict(X[val_inds]))
    return(sum_perf / n_splits)

def create_svc(p):
    svc = SVC(C = p["C"],
              gamma = p["gamma"],
              coef0 = p["coef0"],
              kernel = p["kernel"]["type"],
              degree = p["kernel"]["degree"] if p["kernel"]["type"] == "poly" else 0)
    return(svc)


def tune_tree(tree, X, y):
    if not getattr(tree, 'terminal', False):
       tX, ty = select_data(X, y, tree.classA, tree.classB)
       params = LHS_params(param_list, map_rbf_SVC_sample, count=50)
       tuner = ArrayTuner.ArrayTuner(params)
       for p in tuner:
           tuner.report([p], inverse_KFold(tX, ty, create_svc(p)))
       setattr(tree, "entity", create_svc(tuner.best[0]))
       tune_tree(tree.left, X, y)
       tune_tree(tree.right, X, y)
    return

def tune_all_decisions(args, X, y):
    infile = open(args.structure, 'r')
    tree = make_tree(args.structure)
    print(type(tree))
    tune_tree(tree, X, y)
    #tree.to_json(args.output)
    return(tree)

if __name__ == "__main__":
    args = parse_args()
    X, y, test_X, test_y = load_all_data(args)
    tree = tune_all_decisions(args, X, y)
    tree.fit(X, y)
    outj = json.dumps(tree, cls=TH.TreeEncoder, indent=3)
    print(outj)
    outfile = open(args.output, 'w')
    outfile.write(outj)
    outfile.close
