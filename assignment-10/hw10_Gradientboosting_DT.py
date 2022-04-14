#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 10 18:02:35 2022

@author: milos
"""
import sys, os, argparse
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

from sklearn.ensemble import GradientBoostingClassifier

from classify import (
	print_config, evaluate_classifier, plot_classifier, write_classifier
)

# configure run

print('Configuring run')

parser = argparse.ArgumentParser()
parser.add_argument('--name', default=None)
parser.add_argument('--n_trees', default=5, type=int)
parser.add_argument('--option', default=1, type=int)
parser.add_argument('--data_dir', default='.')
parser.add_argument('--out_dir', default='.')
args = parser.parse_args()

if args.name is None:
	args.name = f'DT{args.option}-Grad{args.n_trees}'

model_kws = [
	dict(min_samples_leaf=5, max_depth=None),
	dict(min_samples_leaf=1, max_depth=None),
	dict(min_samples_leaf=1, max_depth=1)
][args.option-1]

print_config(**vars(args))

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

# model training

print('\nFitting gradient boosting classifier')
model = GradientBoostingClassifier(
	**model_kws, n_estimators=args.n_trees, random_state=0
)
model.fit(X_train, y_train)

#print(f'Node count = {model.tree_.node_count}')
print(f'Train score = {model.score(X_train, y_train)}')
print(f'Test score = {model.score(X_test, y_test)}')

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
