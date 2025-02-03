import numpy as np
from joblib import dump, load
from sklearn.metrics import classification_report as CR
import sys
sys.path.append('../../src')
import treeHierarchy as th
import json
from matplotlib import pyplot as plt

targets = ['cylinder', 'disk', 'multilayer_vesicle', 'core_shell_cylinder']
infile = open("4_class_w_multilayer.json")
jstr = infile.read()
inj = json.loads(jstr)
tree = th.treeHierarchy()
tree.from_json(inj)
datadir = '../../src'
trainX = np.log10(np.loadtxt('%s/smallX_trimmed.csv'%(datadir), delimiter=','))
trainy = np.loadtxt('%s/smally_trimmed.csv'%(datadir), dtype=str)
testX = np.log10(np.loadtxt('%s/testX_trimmed.csv'%(datadir), delimiter=','))
testy = np.loadtxt('%s/testy_trimmed.csv'%(datadir), dtype=str)
tree.fit(trainX, trainy)
preds = tree.predict(testX)
missed_inds = np.where(np.logical_not(preds==testy))[0]
print('MISSED')
print(testy[missed_inds])
print(preds[missed_inds])
print(np.sum(preds==testy))
print(preds)
print(CR(testy, preds))
#inexpcurve = np.loadtxt('../expinter.csv', delimiter=',')
inexpcurve = np.loadtxt('../extrap-1.csv', delimiter=',')
#expcurve = inexpcurve - np.min(inexpcurve) + 0.001
expcurve = inexpcurve
expcurve = np.log10(expcurve)
exp1 = tree.predict(expcurve.reshape(1,-1))
#inexpcurve2 = np.loadtxt('../expinter2.csv', delimiter=',')
inexpcurve2 = np.loadtxt('../extrap-2.csv', delimiter=',')
#expcurve2 = inexpcurve2 - np.min(inexpcurve2) + 0.001
expcurve2 = inexpcurve2
expcurve2 = np.log10(expcurve2)
exp2 = tree.predict(expcurve2.reshape(1,-1))
inexpcurve3 = np.loadtxt('../extrap-3.csv', delimiter=',')
#inexpcurve3 = np.loadtxt('../expinter3.csv', delimiter=',')
#expcurve3 = inexpcurve3 - np.min(inexpcurve3) + 0.001
expcurve3 = inexpcurve3
expcurve3 = np.log10(expcurve3)
exp3 = tree.predict(expcurve3.reshape(1,-1))
print('Experimental %s %s %s'%(exp1, exp1, exp3))


plt.xscale('log')
q = np.loadtxt('../../src/q_trimmed.txt')
#plt.yscale('log')
plt.scatter(q,expcurve,label = 'curve1')
plt.scatter(q,expcurve2,label = 'curve2')
plt.scatter(q,expcurve3,label = 'curve3')
for target in np.unique(trainy):
    match= np.random.choice(np.where(trainy==target)[0])
    plt.scatter(q, trainX[match,:], label = target)
plt.legend()
plt.show()
    
