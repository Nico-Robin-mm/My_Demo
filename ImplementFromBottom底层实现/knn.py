# -*- coding: utf-8 -*-
"""
Created on Tue May 26 02:58:30 2020

@author: 71020
"""


import numpy as np
from collections import Counter
from .metrics import accuracy_score

class knn_classifier:
    
    def __init__(self, k):
        
        assert k >= 1, "k must be valid"
        
        self.k = k 
        self._X_train = None
        self._y_train = None
        
    def fit(self, X_train, y_train):
        
        assert X_train.shape[0] == y_train.shape[0], \
            "the size of X_train and y_train must be same"
            
        self._X_train = X_train
        self._y_train = y_train
        return self
    
    def predict(self, X_predict):
        
        assert X_predict.shape[1] == self._X_train.shape[1], \
            "the features of X_predict and X_train must be equal"
            
        y_predict = [self._predict(x) for x in X_predict]
        return np.array(y_predict)
    
    def _predict(self, x):
        
        assert x.shape[0] == self._X_predict[1], \
            "the features of x and X_train must be same"
            
        distances = [np.sqrt((x - x_train).dot(x - x_train)) for x_train in self._X_train]
        nearest_index = np.argsort(distances)
        topk_y = [self._y_train[i] for i in nearest_index[:self.k]]
        votes = Counter(topk_y)
        return votes.most_common(1)[0][0]
    
    
        
    def score(self, X_test, y_test):
        
        y_predict = self.predict(X_test)
        return accuracy_score(y_test, y_predict)
    
    
    def __repr__(self):
        
        return "my_knn(%i)" %self.k
        
        
        