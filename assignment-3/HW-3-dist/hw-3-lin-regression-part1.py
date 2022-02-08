#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  3 18:17:41 2022

@author: milos
"""

# Problem 2. Part 1. Linear regression

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model, preprocessing
from sklearn.metrics import mean_squared_error

# load the boston housing dataset
housing = datasets.fetch_openml(name='boston',version=1)
housing.details 
X=housing.data
Y=housing.target

# train/test split
X_train=X[:-100]
X_test=X[-100:]
Y_train=Y[:-100]
Y_test=Y[-100:]

# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(X_train, Y_train)
# Print regression coefficients
print('Coefficients: \n', regr.coef_)


# Make predictions on the testing set
Y_test_pred = regr.predict(X_test)
# The mean squared error on the test set
print('Mean squared testing error: %.2f'
      % mean_squared_error(Y_test, Y_test_pred))


       
                    
