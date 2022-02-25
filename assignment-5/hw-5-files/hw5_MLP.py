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
from sklearn.neural_network import MLPClassifier

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
parser.add_argument('-n', '--name', default='mlp')
parser.add_argument('-H', '--hidden_layers', default='4', type=int_tuple)
parser.add_argument('-A', '--activation', default='l', type=get_activ)
parser.add_argument('-s', '--solver', default='adam')
parser.add_argument('-a', '--alpha', default=1e-4, type=float)
parser.add_argument('-l', '--learning_rate', default='c', type=get_lr)
parser.add_argument('-i', '--learning_rate_init', default=1e-3, type=float)
parser.add_argument('-m', '--max_iter', default=1000, type=int)
parser.add_argument('-b', '--batch_size', default=64, type=int)
parser.add_argument('-r', '--random_state', default=None, type=int)
parser.add_argument('-v', '--verbose', default=True, action='store_true')
parser.add_argument('-d', '--data_dir', default='.')

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

print('Fitting MLP classifier')
model = MLPClassifier(
	hidden_layer_sizes=args.hidden_layers,
	activation=args.activation,
	solver=args.solver,
	alpha=args.alpha,
	learning_rate=args.learning_rate,
	learning_rate_init=args.learning_rate_init,
	max_iter=args.max_iter,
	batch_size=args.batch_size,
	random_state=args.random_state,
	verbose=args.verbose
)
model.fit(X_train, y_train)

# model evaluation

print('\nTrain evaluations')
yh_train, pr_train = evaluate_classifier(X_train, y_train, model)

print('\nTest evaluations')
yh_test, pr_test = evaluate_classifier(X_test, y_test, model)

plot_classifier(
	f'plots/{args.name}_evals.png',
	y_train, yh_train, pr_train,
	y_test,  yh_test,  pr_test
)

write_classifier(
	f'metrics/{args.name}_evals.csv',
	['train']*len(y_train) + ['test'] * len(y_test),
	list(y_train)  + list(y_test),
	list(yh_train) + list(yh_test),
	list(pr_train) + list(pr_test)
)
