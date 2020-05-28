# -*- coding: utf-8 -*-
"""
Created on Tue May 26 03:21:12 2020

@author: 71020
"""


import numpy as np


def accuracy_score(y_true, y_predict):
    
    assert y_true.shape[0] == y_predict.shape[0], \
        "the size of y_true and y_predict must be equal"
        
    return sum(y_true == y_predict) / len(y_true)


def mean_squared_error(y_true, y_predict):
    return (y_true - y_predict).dot(y_true - y_predict) / len(y_true)


def root_mean_squared_error(y_true, y_predict):
    return np.sqrt((y_true - y_predict).dot(y_true - y_predict) / len(y_true))


def mean_absolute_error(y_true, y_predict):
    return np.sum(np.absolute(y_true - y_predict)) / len(y_true)


def r2_score(y_true, y_predict):
    return 1 - mean_squared_error(y_true, y_predict) / np.var(y_true)
