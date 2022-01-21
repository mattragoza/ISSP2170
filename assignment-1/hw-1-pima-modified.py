#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 20 13:05:14 2022

@author: milos
"""

# Homework assignment 1 files
# pima dataset problems
  
import numpy as np
import pandas
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import seaborn as sns
sns.set_context('paper')
sns.set_palette('pastel')

# load pima dataset 
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"

names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
df = pandas.read_csv(url, names=names)

X = df.values[:,0:8]
y = df.values[:,8]
print('X shape:', X.shape)
print('y shape:', y.shape)

print('\nFirst few rows:')
print(df.head())

def print_table(df, use_latex=False):
	print(df.to_latex(float_format="{:0.2f}".format) if use_latex else df)

# Problem 2a
print('\nMin and max:')
print_table(df.agg([np.min, np.max]))

# Problem 2b
print('\nMean and std:')
print_table(df.agg([np.mean, np.std]))

# Problem 2c
print('\nMean and std by class:')
print_table(df.groupby('class').agg([np.mean, np.std]).stack())

# problem 2d and 2e
print('\nCorrelation matrix:')
# same output as np.corrcoef(values, rowvar=False)
print_table(df.corr())

def plot_hist(fname, hue=None):
	fig, axes = plt.subplots(4, 2, figsize=(6,8))
	for x, ax in zip(df.columns, axes.flatten()):
		sns.histplot(df, x=x, hue=hue, bins=20, ax=ax)
		if hue:
			ax.get_legend().get_frame().set_alpha(0)
		ax.set_xlabel(x)
		ax.set_ylabel('count')
	sns.despine(fig)
	fig.tight_layout()
	fig.savefig(fname, bbox_inches='tight', dpi=400)

# problem 2f
print('\nPlotting histograms')
plot_hist('plots/hist_plots.png')

# problem 2e
print('Plotting histograms by class')
plot_hist('plots/class_hist_plots.png', hue='class')

# problem 2g
def scatter_plot(fname, xs, ys):
	fig, axes = plt.subplots(len(ys), len(xs), figsize=(6,6))
	for i, y in enumerate(ys):
		for j, x in enumerate(xs):
			ax = axes[i,j]
			ax.scatter(x=df[x], y=df[y], alpha=0.2, marker='+', linewidth=1, s=10)
			if i+1 == len(ys):
				ax.set_xlabel(x)
			else:
				ax.xaxis.set_visible(False)
			if j == 0:
				ax.set_ylabel(y)
			else:
				ax.yaxis.set_visible(False)
	sns.despine(fig)
	fig.tight_layout()
	plt.subplots_adjust(wspace=0.5, hspace=0.5)
	fig.savefig(fname, bbox_inches='tight', dpi=400)

xs = df.columns[:-1]
print('Plotting scatter plots')
scatter_plot('plots/scatter_plots.png', xs, xs)

# problem 3b
print('Normalizing values')
scaler = preprocessing.StandardScaler()
X = scaler.fit_transform(X)
print(X.shape)
print(X[:5,3])

# problem 3c
print('Discretizing values')
discretizer = preprocessing.KBinsDiscretizer(
	n_bins=10, strategy='uniform', encode='onehot-dense', 
)
X = discretizer.fit_transform(X)
print(X.shape)
print(X[:5,30:40])

# problem 4a
print('\nSplitting train and test')
X_train, X_test, y_train, y_test = train_test_split(
	X, y, test_size=0.33, random_state=7
)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# problem 4b
X_train, X_test, y_train, y_test = train_test_split(
	X, y, test_size=0.33, random_state=3
)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# problem 4c
X_train, X_test, y_train, y_test = train_test_split(
	X, y, test_size=0.25, random_state=7
)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
