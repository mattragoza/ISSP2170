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

import seaborn as sns
sns.set_context('paper')
true_color = sns.color_palette('Blues', 3)[2]
pred_colors = sns.color_palette('Greens', 3)

from kde import (
	parzen_window_pdf,
	gaussian_kernel_pdf,
	knn_estimate_pdf
)

# plot settings
dpi = 400
fig_h = 2.5
fig_w = 6.5/3
x_min = 0
x_max = 1

def save_plot(plot_file, fig):
	'''
	Standardize and save a plot figure.
	'''
	for i, ax in enumerate(fig.get_axes()):
		ax.set_xlim(x_min, x_max)
		ax.set_ylim(0, ax.get_ylim()[1])
		ax.set_xticks(np.arange(x_min, x_max+0.25, 0.25))
		ax.set_xlabel('$x$')
		if i == 0:
			ax.set_ylabel('$p(x)$')
		ax.set_axisbelow(True) # grid lines behind data
		ax.grid(True, linestyle=':', color='lightgray')
		ax.legend(frameon=False)
	sns.despine(fig)
	fig.tight_layout()
	print(f'Writing {plot_file}')
	fig.savefig(plot_file, dpi=dpi, bbox_inches='tight')


## Part a - load and plot data

# load the true pdf values on interval [0,1]
true_df = pd.read_csv("true_pdf.csv")
x = true_df.x
pdf = true_df.pdf
print(true_df.describe().transpose())

# plot the true distribution
fig, ax = plt.subplots(figsize=(3, 3))
ax.plot(x, pdf, color=true_color)
ax.set_title('True distribution')
save_plot('true_dist.png', fig)

# load the training data
data_df = pd.read_csv("data-NDE.csv")
D = data_df.x_data
print(data_df.describe().transpose())

# plot the training data
fig, ax = plt.subplots(figsize=(3, 3))
ax.hist(D, bins=10, color=pred_colors[1])
ax.set_title('Data distribution')
save_plot('data_dist.png', fig)

## Part b - define Parzen window pdf

# see kde.py for pdf function definitions

## Part c - plot Parzen window pdf

fig, axes = plt.subplots(1, 3, figsize=(fig_w*3, fig_h))
ax = axes[0]
for i, h in enumerate([0.02, 0.05, 0.10]):
	y = parzen_window_pdf(x, D, h)
	ax.plot(x, y, label=f'h = {h}', color=pred_colors[i])
ax.plot(x, pdf, label='true', color=true_color)
ax.set_title('Parzen window KDE')

## Part d.1 - Gaussian kernel pdf

ax = axes[1]
for i, h in enumerate([0.1, 0.16, 0.25]):
	y = gaussian_kernel_pdf(x, D, h)
	ax.plot(x, y, label=f'h = {h}', color=pred_colors[i])
ax.plot(x, pdf, label='true', color=true_color)
ax.set_title('Gaussian KDE')

## Part d.2 - KNN estimate pdf

ax = axes[2]
for i, k in enumerate([1, 3, 5]):
	y = knn_estimate_pdf(x, D, k)
	ax.plot(x, y, label=f'k = {k}', color=pred_colors[i])
ax.plot(x, pdf, label='true', color=true_color)
ax.set_title('KNN density estimate')

save_plot('kde_dists.png', fig)
