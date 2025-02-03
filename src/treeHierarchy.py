import numpy as np
from sklearn.svm import SVC
import json
from joblib import dump, load

class treeHierarchy:

    def _init_(self):
        self.left = None
        self.right = None
        self.content = None
        self.terminal = None
        self.classA = None
        self.classB = None
    
    def add_content(self, contant, terminal):
        self.content = content
        self.terminal = terminal

    def add_left(self, entity):
        self.left = content

    def add_right(self, entity):
        self.right = content

    def fit(self, X, y):
        print("fitting ")
        print(np.unique(y))
        if not getattr(self, 'terminal', False):
           print('Training [%s] vs. [%s]'%(' '.join(self.classA), ' '.join(self.classB)))
           templabels = np.zeros(X.shape[0])
           for l in np.unique(self.classB):
               for i in range(len(y)):
                   if y[i] == l:
                       templabels[i] = 1
           print(y)
           print(np.sum(templabels))
           print(templabels)
           self.entity.fit(X, templabels)
           ia = np.where(templabels == 0)[0]
           print('IA %d'%(len(ia)))
           ib = np.where(templabels == 1)[0]
           print('IB %d'%(len(ib)))
           self.left.fit(X[ia], y[ia])
           self.right.fit(X[ib], y[ib])
        return

    def predict(self, X):
        print(X.shape)
        if X.shape[0] == 0:
            y = np.zeros(0)
        elif not getattr(self, 'terminal', False):
            print('predicting [%s] vs [%s]'%(' '.join(self.classA), ' '.join(self.classB)))
            inds = np.arange(X.shape[0])
            temp_y = self.entity.predict(X)
            print('TEMP Y')
            print(temp_y)
            print('ENTITY')
            print(self.entity)
            ia = np.where(temp_y == 0)[0]
            ib = np.where(temp_y == 1)[0]
            return_y = np.empty(X.shape[0], dtype=object)
            y_a = self.left.predict(X[ia])
            y_b = self.right.predict(X[ib])
            for v in range(ia.shape[0]):
                vi = ia[v]
                return_y[vi] = y_a[v]
            for v in range(ib.shape[0]):
                vi = ib[v]
                return_y[vi] = y_b[v]
            y = return_y
        else:
            y = np.array([self.terminal] * X.shape[0])
        return(y)

    def from_json(self, J):
        if 'class' in J.keys():
            self.terminal = J['class']
        else:
            if 'jobfile' in J.keys():
                self.entity = load(J['jobfile'])
                print('Loading %s'%(J['jobfile']))
            else:
                self.entity = classifier_from_json(J['classifier'])
            self.left = treeHierarchy()
            self.left.from_json(J['left'])
            self.right = treeHierarchy()
            self.right.from_json(J['right'])
            self.classA = J['classLeft']
            self.classB = J['classRight']

def classifier_from_json(J):
    if J['type'] == 'svc':
        K = J['kernel']
        classifier = SVC(C = J['c'],
        gamma = J['gamma'],
        kernel = K['type'],
        degree = K['degree'] if K['type'] == 'polynomial' else 1,
        coef0 = J['coeff0'])
    else:
        classifier = None
    return(classifier)
