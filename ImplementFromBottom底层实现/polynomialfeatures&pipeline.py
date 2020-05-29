# -*- coding: utf-8 -*-
"""
Created on Fri May 29 22:25:59 2020

@author: 71020
"""


import numpy as np
import matplotlib.pyplot as plt


x = np.random.uniform(-3, 3, size=100)
X = x.reshape(-1, 1)
y = 0.5 * x**2 + x + 2 +np.random.normal(0, 1, 100)

from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree=2)
poly.fit(X)
X2 = poly.transform(X)
print(X2.shape)

# =============================================================================
# or
# =============================================================================

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

poly_reg = Pipeline([
    ("poly", PolynomialFeatures(degree=2)),
    ("std_scaler", StandardScaler()),
    ("lin_reg", LinearRegression())
    ])
poly_reg.fit(X, y)
y_predict = poly_reg.predict(X)
