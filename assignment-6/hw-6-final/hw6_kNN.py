#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 10 18:02:35 2022

@author: milos
"""

import sys, os, argparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

from classify import (
	print_config, evaluate_classifier, plot_classifier, write_classifier
)

print('Configuring run')

def int_tuple(s):
	return tuple(int(x) for x in s.split('-'))

def get_activ(a):
	return dict(l='logistic', t='tanh', r='relu')[a]

def get_lr(l):
	return dict(c='constant', i='invscaling', a='adaptive')[l]

# command line args
parser = argparse.ArgumentParser()
parser.add_argument('-n', '--name', default='knn')
parser.add_argument('-k', '--n_neighbors', default=5, type=int)
parser.add_argument('-w', '--weights', default='uniform')
parser.add_argument('-d', '--data_dir', default='.')
parser.add_argument('-o', '--out_dir', default='.')

args = parser.parse_args()
print_config(args)

# load data sets

print('\nLoading training set')

df_train = pd.read_csv(os.path.join(args.data_dir, 'pima_train.csv'))
print(df_train.head())

X_train = df_train.values[:,0:8]
y_train = df_train.values[:,8]
print(X_train.shape, y_train.shape)

print('\nLoading test set')

df_test = pd.read_csv(os.path.join(args.data_dir, 'pima_test.csv'))
print(df_test.head())

X_test = df_test.values[:,0:8]
y_test = df_test.values[:,8]
print(X_test.shape, y_test.shape)

# preprocessing

print('\nStandardizing input variables')

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)  
X_test = scaler.transform(X_test)

# model training

print('Fitting k-NN classifier')
model = KNeighborsClassifier(
	n_neighbors=args.n_neighbors,
	weights=args.weights
)
model.fit(X_train, y_train)

# model evaluation

print('\nTrain evaluations')
yh_train, pr_train = evaluate_classifier(X_train, y_train, model)

print('\nTest evaluations')
yh_test, pr_test = evaluate_classifier(X_test, y_test, model)

plot_classifier(
	f'{args.out_dir}/{args.name}_evals.png',
	y_train, yh_train, pr_train,
	y_test,  yh_test,  pr_test
)

write_classifier(
	f'{args.out_dir}/{args.name}_evals.csv',
	['train']*len(y_train) + ['test'] * len(y_test),
	list(y_train)  + list(y_test),
	list(yh_train) + list(yh_test),
	list(pr_train) + list(pr_test)
)
