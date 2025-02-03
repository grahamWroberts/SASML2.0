import numpy as np
from joblib import dump, load
from sklearn.metrics import classification_report as CR

targets = ['cylinder', 'disk', 'sphere', 'cs_cylinder', 'cs_disk', 'cs_sphere']
th = load('whole_tree.joblib')
testX = np.loadtxt('testX.csv')
testy = np.loadtxt('testy.csv')
true = np.array([targets[int(i)] for i in testy])
preds = th.predict(testX)
print(CR(true, preds))
