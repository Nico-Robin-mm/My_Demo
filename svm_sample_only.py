# -*- coding: utf-8 -*-
"""
Created on Sun May 31 03:40:38 2020

@author: 71020
"""


from sklearn import datasets


iris = datasets.load_iris()
X = iris.data
y = iris.target

X = X[y<2, :2]
y = y[y<2]

from sklearn.preprocessing import StandardScaler

standardScaler = StandardScaler()
standardScaler.fit(X)
X_standard = standardScaler.transform(X)

from sklearn.svm import LinearSVC

svc = LinearSVC(C=1e9) # hard svm
svc2 = LinearSVC(C=1) # soft svm
svc.fit(X_standard, y)
svc.coef_ 
svc.intercept_ 

