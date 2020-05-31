# -*- coding: utf-8 -*-
"""
Created on Sun May 31 16:50:01 2020

@author: 71020
"""


from sklearn.decomposition import PCA
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

digits = datasets.load_digits()
X = digits.data 
y = digits.target

X_train, X_test, y_train, y_test = train_test_split(X, y)

pca = PCA(n_components=2)
pca.fit(X_train)
X_train_reduction = pca.transform(X_train)
X_test_reduction = pca.transform(X_test)

knn_clf = KNeighborsClassifier()
knn_clf.fit(X_train_reduction, y_train)
print(knn_clf.score(X_test_reduction, y_test))

# info amount of chosen components

print("包含了多少方差的信息，即原数据的信息量:", pca.explained_variance_ratio_)

pca2 = PCA(n_components=X_train.shape[1])
pca2.fit(X_train)
print("pca2 info amount:", pca2.explained_variance_ratio_)
plt.plot([i for i in range(X_train.shape[1])],
         [np.sum(pca2.explained_variance_ratio_[:i+1]) for i in range(X_train.shape[1])])
plt.show()

# valid decomposition

pca3 = PCA(0.95)
pca3.fit(X_train)
print(pca3.n_components_)
X_train_reduction3 = pca3.transform(X_train)
X_test_reduction3 = pca3.transform(X_test)
knn_clf3 = KNeighborsClassifier()
knn_clf3.fit(X_train_reduction3, y_train)
print(knn_clf3.score(X_test_reduction3, y_test))

# visible

pca4 = PCA(n_components=2)
pca4.fit(X)
X_reduction = pca4.transform(X)
print(X_reduction.shape)
for i in range(10):
    plt.scatter(X_reduction[y==i, 0], X_reduction[y==i, 1], alpha=0.8)
plt.show()