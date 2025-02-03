import sys
sys.path.append('.')
import json
import treeHierarchy as TH
import numpy as np
from sklearn.metrics import classification_report as CR
from joblib import dump, load

targets = ['cylinder', 'disk', 'sphere', 'cs_cylinder', 'cs_disk', 'cs_sphere']


def print_th(th):
    if not getattr(th, 'terminal', False):
       print('classifier: ')
       try:
          print(th.entity)
       except:
          print('no entity')
          print('TERMINAL: %s\n'%(th.terminal))
       #print('\n')
       print('left %s'%(' '.join(th.classA)))
       print_th(th.left)
       print('right %s'%(' '.join(th.classB)))
       print_th(th.right)
    else:
       print('TERMINAL: %s'%(th.terminal))

th = TH.treeHierarchy()
print(th)
print(getattr(th, 'terminal', False))
jfile = open('ex_hierarchical.json', 'r')
inj = json.load(jfile)
th.from_json(inj)
print_th(th)
trainX = np.loadtxt('trainX.csv')
trainy = np.loadtxt('trainy.csv')
labels = np.array([targets[int(i)] for i in trainy])
th.fit(trainX, labels)
testX = np.loadtxt('testX.csv')
testy = np.loadtxt('testy.csv')
preds = th.predict(testX)
testy = np.loadtxt('testy.csv')

true = np.array([targets[int(i)] for i in testy])
print(preds)
print(true)
print(CR(true, preds))
dump(th, 'whole_tree.joblib')


