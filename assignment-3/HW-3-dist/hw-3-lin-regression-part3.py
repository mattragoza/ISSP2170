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

import seaborn as sns
sns.set_context('paper')
true_color = sns.color_palette('Blues', 3)[2]
pred_colors = sns.color_palette('Greens', 3)

# plot settings
dpi = 400
fig_h = 3
fig_w = 5

def save_plot(plot_file, fig):
	'''
	Standardize and save a plot figure.
	'''
	for i, ax in enumerate(fig.get_axes()):
		ax.set_ylim(0, 1000)
		ax.set_axisbelow(True) # grid lines behind data
		ax.grid(True, linestyle=':', color='lightgray')
		ax.legend(frameon=False)
	sns.despine(fig)
	fig.tight_layout()
	print(f'Writing {plot_file}')
	fig.savefig(plot_file, dpi=dpi, bbox_inches='tight')


class LinearRegression(object):

	def __init__(self, ax=None):
		self.W = None
		self.ax = ax

	def fit(
		self, X, Y,
		shuffle=True,
		batch=True,
		n_iters=1000,
		lr_fn=lambda x: 0.0001,
		init_fn=lambda x: np.random.normal(0, 1, x)
	):
		print('fitting model', flush=True)

		# convert to arrays
		X = np.array(X)
		Y = np.array(Y)

		# check data shapes
		N, D = X.shape
		assert Y.shape == (N,)
		Y = Y.reshape(N, 1)

		# add intercept term
		X = np.concatenate((np.ones((N, 1)), X), axis=1)
		D += 1

		if shuffle: # randomize data order
			order = np.random.permutation(N)
			X = X[order]
			Y = Y[order]

		# initialize coefficients
		self.W = init_fn((D, 1))
		print(X.shape, self.W.shape, Y.shape)

		iters = np.arange(n_iters+1)
		MSE = np.full(n_iters+1, np.nan)
		for i in iters:

			if batch: # batch mode
				if shuffle:
					order = np.random.permutation(N)
					X_curr = X[order]
					Y_curr = Y[order]
				else:
					X_curr = X
					Y_curr = Y
			else: # online mode
				j = i%N
				X_curr = X[j:j+1]
				Y_curr = Y[j:j+1]

			# compute predictions and error
			Y_pred = np.matmul(X_curr, self.W)
			Y_diff = Y_curr - Y_pred
			MSE[i] = (Y_diff**2).mean()
			lr = lr_fn(i+1)
			print(f'Iteration {i}, MSE = {MSE[i]:.2f}, lr = {lr:.6f}')

			if i == n_iters:
				break

			# compute gradient and update
			grad_W = -2 * X_curr.T @ Y_diff
			self.W = self.W - lr * grad_W

		if self.ax:
			self.ax.plot(iters, MSE)

	def predict(self, X):

		# add intercept term
		N, D = X.shape
		X = np.concatenate((np.ones((N, 1)), X), axis=1)
		return X @ self.W


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

# train/test split
X_train = X[:-100]
X_test = X[-100:]
Y_train = Y[:-100]
Y_test = Y[-100:]

# standardize the predictors
scaler = preprocessing.StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# create training plot
fig, ax = plt.subplots(figsize=(fig_w, fig_h))

# create linear regression model
model = LinearRegression(ax=ax)

# train the model using the training set
model.fit(X_train, Y_train)

ax.set_title(np.random.rand())
save_plot('training.png', fig)

# print regression coefficients
print('Coefficients: \n', model.W[:,0])

# Make predictions on the train and test sets
Y_train_pred = model.predict(X_train)
Y_test_pred = model.predict(X_test)

# The mean squared error on the train and tests set
MSE_train = mean_squared_error(Y_train, Y_train_pred)
MSE_test = mean_squared_error(Y_test, Y_test_pred)

print(f'Train MSE: {MSE_train:.2f}')
print(f'Test MSE:  {MSE_test:.2f}')
print(f'{(MSE_test-MSE_train)/MSE_train:.3f}')
