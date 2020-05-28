# -*- coding: utf-8 -*-
"""
Created on Thu May 28 01:51:42 2020

@author: 71020
"""


import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV


digits = datasets.load_digits()
X = digits.data
y = digits.target
X_train, X_test, y_train, y_test = train_test_split(X, y)
best_k, best_p, best_score = 0, 0, 0

for k in range(2, 11):
    for p in range(1, 6):
        knn_clf = KNeighborsClassifier(n_neighbors=k, weights="distance", p=p)
        scores = cross_val_score(knn_clf, X_train, y_train)
        score = np.mean(scores)
        if score > best_score:
            best_score = score
            best_k = k
            best_p = p
print("best_k:", k)
print("best_p:", p)
print("best_score:", score)

# =============================================================================
# or
print("======================================================================")
# =============================================================================

param_grid = [
    {
     "weights": ["distance"],
     "n_neighbors": [i for i in range(2, 11)],
     "p": [i for i in range(1, 6)]
     }
    ]
grid_search = GridSearchCV(knn_clf, param_grid, verbose=1)
grid_search.fit(X_train, y_train)
print("best_score of grid_search:", grid_search.best_score_)
print("best_params of grid_search:", grid_search.best_params_)
best_knn_clf = grid_search.best_estimator_
print(best_knn_clf.score(X_train, y_train))




