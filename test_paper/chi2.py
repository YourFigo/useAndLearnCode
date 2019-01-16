# -*- coding: utf-8 -*-
"""
Created on Tue Jan 15 13:03:02 2019

@author: Figo
"""

import numpy as np

X = np.array([[1,3,5,7],[2,4,6,8],[3,2,6,8]])
y = np.array([[1],[0],[1]])

observed = np.dot(y.T, X)

feature_count = X.sum(axis=0).reshape(1, -1)
class_prob = y.mean(axis=0).reshape(1, -1)
expected = np.dot(class_prob.T, feature_count)

observed = np.asarray(observed, dtype=np.float64)
k = len(observed)
    # Reuse f_obs for chi-squared statistics
chisq = observed
chisq -= expected
chisq **= 2
with np.errstate(invalid="ignore"):
    chisq /= expected
