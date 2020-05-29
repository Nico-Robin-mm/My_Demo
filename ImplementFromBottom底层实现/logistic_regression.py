# -*- coding: utf-8 -*-
"""
Created on Sat May 30 02:05:34 2020

@author: 71020
"""


import numpy as np


class logistic_regression:
    
    def __init__(self):
        self._theta = None
        self.intercept_ = None
        self.coef_ = None
            
    def _sigmoid(z):
        return 1. / (1. + np.exp(-z))
    
    def fit(X_train, y_train, n_iters=1e4, lr=0.01, epsilon=1e-18):
        
        def J(theta, X_b, y):
            y_hat = self._sigmoid(X.dot(theta))
            try:
                return -np.sum(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat)) / len(y)
            except:
                return float("inf")
            
        def dJ(theta, X_b, y):
            return X_b.T.dot(self._sigmoid(X_b.dot(theta)) - y) / len(y)
        
        def gradient_descent(X_b, y, initial_theta, n_iters, lr, epsilon):
            theta = initial_theta 
            last_theta = theta 
            i_iter = 0
            while i_iter < n_iters:
                theta = theta - lr * dJ(theta, X_b,, y)
                if abs(J(last_theta, X_b, y) - J(theta, X_b, y)) < epsilon:
                    break
            return theta
                
        X_b = np.hstack([np.ones(shape=(len(X_train), 1)), X_train])
        initial_theta = np.zeros(shape=X_b.shape[1])
        self._theta = gradient_descent(X_b, y_train, n_iters, lr, epsilon)
        self.intercept_ = self._theta[0]
        self.coef_ = self._theta[1:]
        return self 
    
    def predict_prob(self, X_predict):
        X_b = np.hstack([np.ones(shape=(X_predict.shape[0], 1)), X_predict])
        return self._sigmoid(X_b.dot(self._theta))
    
    def predict(self, X_predict):
        return np.array(self.predict_prob(X_predict) >= 0.5, dtype="int")
    
    def score(self, X_test, y_test):
        y_predict = self.predict(X_test)
        return sum(y_predict == y_test) / len(y_test)
    
    def __repr__(self):
        return "mylogisticregression"