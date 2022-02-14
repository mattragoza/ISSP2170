#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 11 00:42:08 2022

@author: milos
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

### ML Classification models
from sklearn.linear_model import LogisticRegression  # Log reg
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, confusion_matrix
from sklearn.metrics import precision_recall_curve


dataframe = pd.read_csv('pima.csv')
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
targets=['class 0', 'class 1']
