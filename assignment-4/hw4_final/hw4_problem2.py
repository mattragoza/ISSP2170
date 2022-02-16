#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 11 00:42:08 2022

@author: milos
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_context('paper')
sns.set_palette('pastel')

# load data set

print('Loading data set')
df = pd.read_csv('pima.csv')
print(df.head())

feats = df.columns[:-1]
n_feats = len(feats)

print('Descriptive statistics')
pd.set_option('display.max_columns', 10)
print(df.describe().transpose())

# visualze the data

def save_plot(plot_file, fig, axes):

    for ax in axes:
        ax.set_ylabel('count')
        ax.set_axisbelow(True) # grid lines behind data
        ax.grid(True, linestyle=':', color='lightgray')

    sns.despine(fig)
    fig.tight_layout()
    fig.savefig(plot_file, bbox_inches='tight', dpi=400)

# plot class distribution

fig, ax = plt.subplots(figsize=(2.75, 2.75))
bins = np.linspace(-0.25, 1.25, 4)
sns.histplot(data=df, x='class', hue='class', bins=bins, ax=ax, legend=False)
ax.set_xlim(-0.75, 1.75)
ax.set_xticks([0, 1])
save_plot('class_dist.png', fig, [ax])

# plot class-conditional distributions
fig, axes = plt.subplots(4, 2, figsize=(6.5, 8))
axes = axes.flatten()
for ax, x in zip(axes, feats):
    sns.histplot(data=df, bins=20, hue='class', x=x, ax=ax)
    ax.get_legend().get_frame().set_visible(False)

save_plot('class_cond_dists.png', fig, axes)
