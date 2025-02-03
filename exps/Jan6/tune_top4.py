import numpy as np
import sys
sys.path.append('../../src')
import treeHierarchy as TH
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

def make_folds(labels, count=5):
    inds = [None for i in range(count)]
    for t in np.unique(labels):
        matches = np.where(labels==t)[0]
        np.random.shuffle(matches)
        splits = np.linspace(0, len(matches), count+1, dtype=int)
        for i in range(count):
            if inds[i] is None:
                inds[i] = matches[splits[i]:splits[i+1]]
            else:
                inds[i] = np.append(inds[i],matches[splits[i]:splits[i+1]])
    return(inds)

def kfold(X, y, classifier, fold_inds):
    fold_accs = np.zeros(len(fold_inds))
    for i in range(len(fold_inds)):
        train_inds = fold_inds[i]
        val_folds = [j for j in range(len(fold_inds)) if j != i]
        val_inds = fold_inds[val_folds[0]]
        for j in range(1,len(val_folds)):
            val_inds = np.append(val_inds, fold_inds[val_folds[j]])
        classifier.fit(X[train_inds,:], y[train_inds])
        train_preds = classifier.predict(X[train_inds,:])
        print('TRAIN REPORT')
        print(classification_report(y[train_inds], train_preds))
        preds = classifier.predict(X[val_inds,:])
        fold_accs[i] = accuracy_score(y[val_inds], preds)
        print(classification_report(y[val_inds], preds))
    print(fold_accs)
    return(np.mean(fold_accs))


def main():
    X = np.log10(np.loadtxt('../../src/smallX_trimmed.csv', delimiter=',')+0.001)
    #X = np.log10(np.loadtxt('../../src/exampleX.csv', delimiter=',')+0.001)
    y = np.loadtxt('../../src/smally_trimmed.csv', dtype=str)
    #y = np.loadtxt('../../src/exampley.csv', dtype=str)
    outfile = open('performance.csv', 'w')
    newlabs = np.array([0 if l in ['cylinder', 'disk'] else 1 for l in y])
    print(newlabs)
    Cs = [10.0, 100.0, 1000.0]
    gammas = ['auto']
    kernels = ['rbf']
    coeff0s = [0, 1]
    paramlist = []
    for C in Cs:
        for gamma in gammas:
            for kernel in kernels:
                for coeff0 in coeff0s:
                    paramlist += [{'C':C, "gamma":gamma, "kernel":kernel, "coeff0": coeff0}]
    print(paramlist)
    fold_inds = make_folds(newlabs)
    for params in paramlist:
       classifier = SVC(C=params["C"], gamma = params["gamma"], kernel = params["kernel"], coef0 = params["coeff0"])
       acc = kfold(X,newlabs,classifier,fold_inds)
       outstring = "%0.2e %s %s %d %f"%(params["C"], params["gamma"], params["kernel"], params["coeff0"], acc)
       print(outstring)
       outfile.write('%s\n'%(outstring))

if __name__ == '__main__':
    main()
