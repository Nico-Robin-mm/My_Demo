# -*- coding: utf-8 -*-
"""
Created on Thu May 28 03:37:00 2020

@author: 71020
"""


import numpy as np


def train_test_split(X, y, test_ratio=0.25, seed=None):
    
    assert X.shape[0] == y.shape[0], \
        "the size of X and y must be equal"
    assert 0.0 <= test_ratio <= 1.0, \
        "test_ratio must be valid"
    
    if seed:
        np.random.seed(seed)
    
    shuffle_index = np.random.permutation(len(X))
    test_size = int(test_ratio * len(X))
    train_index = shuffle_index[test_size:]
    test_index = shuffle_index[:test_size]
    
    X_train = X[train_index]
    y_train = y[train_index]
    X_test = X[test_index]
    y_test = y[test_index]
    
    return X_train, X_test, y_train, y_test
