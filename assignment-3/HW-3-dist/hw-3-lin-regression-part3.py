#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  3 18:17:41 2022

@author: milos
"""

# Problem 2. Part 3. Stochastic gradient descent

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model, preprocessing
from sklearn.metrics import mean_squared_error

import seaborn as sns
sns.set_context('paper')
colors = sns.color_palette()

from linreg import LinearRegression

# plot settings
dpi = 400
fig_h = 3
fig_w = 6.5

def save_plot(plot_file, fig):
	'''
	Standardize and save a plot figure.
	'''
	axes = fig.get_axes()
	for i, ax in enumerate(axes):
		if i < len(axes)//2: # MSE
			ax.set_ylim(-200, 2000)
			ax.set_ylabel('MSE')
			ax.set_axisbelow(True) # grid lines behind data
			ax.grid(True, linestyle=':', color='lightgray')
			han1, lab1 = ax.get_legend_handles_labels()
		else: # alpha
			ax.set_ylim(-2e-4, 2e-3)
			ax.set_ylabel(' $\\alpha$', rotation=0)
			han2, lab2 = ax.get_legend_handles_labels()
			ax.legend(han1+han2, lab1+lab2)
			ax.get_legend().get_frame().set_linewidth(0)
		ax.set_xlabel('Iteration')
	sns.despine(fig, right=False)
	fig.tight_layout()
	print(f'Writing {plot_file}')
	fig.savefig(plot_file, dpi=dpi, bbox_inches='tight')


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

# create linear regression model
model = LinearRegression()

# choose hyperparameters
hparams_def = dict( # defaults
    shuffle=False,
    batch=False,
    n_iters=1000,
    alpha_fn=lambda i, n_iters: 5e-2/i,
    init_fn=lambda shape: np.ones(shape),
)

# train the model using the training set
print('Fitting default model')
iters, MSE, alpha = model.fit(X_train, Y_train, **hparams_def)

# print regression coefficients
print('Coefficients: \n', model.W[:,0])

# Make predictions on the train and test sets
Y_train_pred = model.predict(X_train)
Y_test_pred = model.predict(X_test)

# The mean squared error on the train and tests set
MSE_train = mean_squared_error(Y_train, Y_train_pred)
MSE_test = mean_squared_error(Y_test, Y_test_pred)
MSE_ratio = MSE_test/MSE_train * 100

print(f'Train MSE: {MSE_train:.2f}')
print(f'Test MSE:  {MSE_test:.2f} ({MSE_ratio:.2f}%)')

# create training plot
fig, axes = plt.subplots(1, 2, figsize=(fig_w, fig_h))
ax = axes[0]
ax.plot(iters, MSE, color=colors[0], label='MSE')
ax.set_xlim(-100, hparams_def['n_iters']+100)

ax = ax.twinx()
ax.plot(iters, alpha, color=colors[1], label='$\\alpha$')
ax.set_title('Default model')

# optimized hyperparameters

hparams_opt = dict( # optimized
    shuffle=True,
    batch=True,
    n_iters=250,
    alpha_fn=lambda i, n_iters: 4e-4 * 0.5**(i/(n_iters+1)*5).astype(int),
    init_fn=lambda shape: np.random.normal(0, 1, shape),
)

# train the model using the training set
print('Fitting optimized model')
iters, MSE, alpha = model.fit(X_train, Y_train, **hparams_opt)

# print regression coefficients
print('Coefficients: \n', model.W[:,0])

# Make predictions on the train and test sets
Y_train_pred = model.predict(X_train)
Y_test_pred = model.predict(X_test)

# The mean squared error on the train and tests set
MSE_train = mean_squared_error(Y_train, Y_train_pred)
MSE_test = mean_squared_error(Y_test, Y_test_pred)
MSE_ratio = MSE_test/MSE_train * 100

print(f'Train MSE: {MSE_train:.2f}')
print(f'Test MSE:  {MSE_test:.2f} ({MSE_ratio:.2f}%)')

ax = axes[1]
ax.plot(iters, MSE, color=colors[0], label='MSE')
ax.set_xlim(-25, hparams_opt['n_iters']+25)

ax = ax.twinx()
ax.plot(iters, alpha, color=colors[1], label='$\\alpha$')
ax.set_title('Optimized model')

save_plot('training.png', fig)
