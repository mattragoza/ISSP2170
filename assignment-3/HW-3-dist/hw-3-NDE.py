#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  3 18:15:17 2022

@author: milos
"""

# Problem 1 - nonparametric density estimation


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# loads the true pdf values on interval [0,1]
dataframe = pd.read_csv("true_pdf.csv")
true_dist = dataframe.values
#plots the true distribution
plt.plot(true_dist[:,0],true_dist[:,1])

# load the training data
datainstances = pd.read_csv("data-NDE.csv")
data=datainstances.values
plt.hist(data,10)