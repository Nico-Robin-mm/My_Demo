# -*- coding: utf-8 -*-
"""
Created on Tue May 26 04:29:44 2020

@author: 71020
"""


import numpy as np


class linear_regression:
    
    def __init__(self):
        self.intercept_ = None
        self.coef_ = None
        self._theta = None
    
    def fit_normal(self, X_train, y_train):
        X_b = np.hstack(np.ones((X_train.shape[0], 1)), X_train)
        self._theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y_train)
        self.intercept_ = self._theta[0]
        self.coef = self._theta[1:]
        return self
    
    def fit_gd(self, X_train, y_train, lr=0.01, n_iters=1e4, epsilon=1e-10):        
        def J(theta, X_b, y):
            try:
                return (y - X_b.dot(theta)).T.dot(y - X_b.dot(theta)) / len(y)
            except:
                return float("inf")       
        def dJ(theta, X_b, y):
            return X_b.T.dot(X_b.dot(theta) - y) * 2 / len(y)        
        def gradient_descent(initial_theta, X_b, y, lr, n_iters, epsilon):
            theta = initial_theta
            i_iters = 0
            while i_iters < n_iters:
                last_theta = theta
                theta = theta - lr * dJ(theta, X_b, y)
                if abs(J(last_theta, X_b, y) - J(theta, X_b, y)) < epsilon:
                    break
                i_iters += 1
            return theta
        X_b = np.hstack([np.ones(shape=(len(X_train), 1)), X_b])
        initial_theta = np.zeros(X_b.shape[1])
        self._theta = gradient_descent(initial_theta, X_b, y_train, lr, n_iters, epsilon)
        self.intercept_ = self._theta[0]
        self.coef_ = self._theta[1:]
        return self        
        
    def fit_sgd(self, X_train, y_train, n_iters=5, t0=5, t1=50):        
        def dJ(theta, X_b_i, y_i):
            return X_b_i.T.dot(X_b_i.dot(theta) - y_i) * 2             
        def sgd(X_b, y, initial_theta, n_iters, t0, t1):
            def learning_rate(t):
                return t0 / (t + t1)
            theta = initial_theta
            m = len(X_b)
            for i_iter in range(n_iters):
                rand_i = np.random.randint(m)
                theta = theta - learning_rate(i_iter) * dJ(theta, X_b[rand_i], y[rand_i])
            return theta
        X_b = np.hstack([np.ones(shape=(len(X_train), 1)), X_train])
        initial_theta = np.zeros(X_b.shape[1])
        theta = sgd(X_b, y_train, initial_theta, n_iters, t0, t1)
        return theta
    
    def predict(self, X_predict):
        X_b = np.hstack([np.ones((len(X_predict))), X_predict])
        return X_b.dot(self._theta)
    
    def score(self, X_test, y_test):
        X_b = np.hstack([np.ones((len(X_test))), X_test])
        return 1 - (y_test - X_b * self._theta).T.dot(y_test - X_b * self._theta) / np.var(y_test)
    
    def __repr__(self):
        return "linear_regression"
    
            
            
            
            
            
            
            
            
            
            
            
            
            
            