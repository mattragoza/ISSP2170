#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  3 18:17:41 2022

@author: milos
"""

# Problem 2. Part 2. Qudaratic regression

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model, preprocessing
from sklearn.metrics import mean_squared_error

from linreg import quadratic_expansion


# load the boston housing dataset
housing = datasets.fetch_openml(name='boston', version=1)
X = housing.data
Y = housing.target

# display data shape, type, and descriptive stats
pd.set_option('display.max_columns', 10)
print(type(X), X.shape)

# convert categorical to numeric
X = X.apply(pd.to_numeric)

print(X.apply(lambda x: x.dtype))
print(X.describe().transpose())

print()
print(type(Y), Y.shape)
print(Y.describe().transpose())

# add quadratic terms
X = quadratic_expansion(X)
print(list(X.columns))
print(X.shape)

# train/test split
X_train = X[:-100]
X_test = X[-100:]
Y_train = Y[:-100]
Y_test = Y[-100:]

# create linear regression model
model = linear_model.LinearRegression()

# train the model using the training set
model.fit(X_train, Y_train)

# print regression coefficients
print('Coefficients: \n', model.coef_)

# Make predictions on the train and test sets
Y_train_pred = model.predict(X_train)
Y_test_pred = model.predict(X_test)

# The mean squared error on the train and tests set
MSE_train = mean_squared_error(Y_train, Y_train_pred)
MSE_test = mean_squared_error(Y_test, Y_test_pred)
MSE_ratio = MSE_test/MSE_train * 100

print(f'Train MSE: {MSE_train:.2f}')
print(f'Test MSE:  {MSE_test:.2f} ({MSE_ratio:.2f}%)')
