# -*- coding: utf-8 -*-
"""
Created on Thu May 28 04:01:25 2020

@author: 71020
"""


import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier


iris = datasets.load_iris()
X = iris.data[:, 2:]
y = iris.target

plt.scatter(X[y==0, 0], X[y==0, 1])
plt.scatter(X[y==1, 0], X[y==1, 1])
plt.scatter(X[y==2, 0], X[y==2, 1])
plt.show()

dt_clf = DecisionTreeClassifier(max_depth=2, criterion="entropy")
dt_clf.fit(X, y)


def plot_decision_boundary(model, axis):
    x0, x1 = np.meshgrid(
        np.linspace(axis[0], axis[1], int((axis[1] - axis[0]) * 100)).reshape(1, -1),
        np.linspace(axis[2], axis[3], int((axis[3] - axis[2]) * 100)).reshape(1, -1)
        )
    X_new = np.c_[x0.ravel(), x1.ravel()]
    y_predict = model.predict(X_new)
    zz = y_predict.reshape(x0.shape)
    from matplotlib.colors import ListedColormap
    custom_cmap = ListedColormap(['#EF9A9A',"#FFF59D","#90CAF9"])
    plt.contourf(x0, x1, zz, cmap=custom_cmap)
plot_decision_boundary(dt_clf, axis=[0.5, 7.5, 0, 3])
plt.scatter(X[y==0, 0], X[y==0, 1])
plt.scatter(X[y==1, 0], X[y==1, 1])
plt.scatter(X[y==2, 0], X[y==2, 1])
plt.show()


def split(X, y, d, value):
    index_a = (X[:, d] <= value)
    index_b = (X[:, d] > value)
    return X[index_a], X[index_b], y[index_a], y[index_b]

from collections import Counter
from math import log
def entropy(y):
    counter = Counter(y)
    res = 0.0 
    for num in counter.values():
        p = num / len(y)
        res += -p * log(p) 
    return res

def try_split(X, y):
    
    best_entropy = float("inf")
    best_d, best_v = -1, -1
    for d in range(X.shape[1]):
        sorted_index = np.argsort(X[:, d])
        for i in range(1, len(X)):
            if X[sorted_index[i-1], d] != X[sorted_index[i], d]:
                v = (X[sorted_index[i-1], d] + X[sorted_index[i], d]) / 2
                X_l, X_r, y_l, y_r = split(X, y, d, v)
                e = entropy(y_l) + entropy(y_r)
                if e < best_entropy:
                    best_entropy, best_d, best_v = e, d, v
    return best_entropy, best_d, best_v

best_entropy, best_d, best_v = try_split(X, y)
print("best_entropy=", best_entropy)
print("best_d=", best_d)
print("best_v=", best_v)

X1_l, X1_r, y1_l, y1_r = split(X, y, best_d, best_v)
print(entropy(y1_l))
print(entropy(y1_r))

best_entropy2, best_d2, best_v2 = try_split(X1_r, y1_r)
print("best_entropy2=", best_entropy2)
print("best_d2=", best_d2)
print("best_v2=", best_v2)

X2_l, X2_r, y2_l, y2_r = split(X1_r, y1_r, best_d2, best_v2)
print(entropy(y2_l))
print(entropy(y2_r))














