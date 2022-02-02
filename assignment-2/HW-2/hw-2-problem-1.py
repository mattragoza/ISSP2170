#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 27 08:50:08 2022

@author: milos
"""

#### stats basics: calculate sample mean, standard error and a confidence interval

## sample
import os
import numpy as np
import pandas as pd
import scipy.stats
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_context('paper')
sns.set_palette('pastel')
np.random.seed(1)

fig_w = 6.5
fig_h = 2
y_max = 175

#### the function calculates basic sample stats: sample mean, standard error and a confidence interval
def confidence_interval(sample, confidence_level):
    # the function calculates the sample mean, sample standard error, and confidence interval
    degrees_freedom = sample.size - 1
    sample_mean = np.mean(sample)
    sample_standard_error = scipy.stats.sem(sample)
    # compute confidence interval for `sample`
    confidence_interval = scipy.stats.t.interval(confidence_level, degrees_freedom, sample_mean, sample_standard_error)
    print('Sample mean:', sample_mean)
    print('Sample standard error:', sample_standard_error)
    print('Confidence interval:',confidence_interval)
    return sample_mean, sample_standard_error, confidence_interval

print(os.getcwd())

# loading the data from the text file
data = pd.read_csv('mean_study_data.txt')
sample = data.values[:,0]

# part 1 - sample mean and std
print('Overall sample statistics')
print(f'mean = {np.mean(sample)}')
print(f'std  = {np.std(sample)}')

# part 2 - implement subsample
def subsample(data, k):
    '''
    Randomly select k instances
    from the provided sample.
    '''
    data = data.copy()
    np.random.shuffle(data)
    return data[:k]

# part 3 - plot means of 1000 subsamples
samples = np.stack(
    [subsample(sample, k=25) for i in range(1000)]
)
sample_means = np.mean(samples, axis=1)
sample_std   = np.std(samples, axis=1)

fig, axes = plt.subplots(1, 2, figsize=(fig_w, fig_h))
ax = axes[0]
sns.histplot(sample_means, bins=np.linspace(12, 18, 21), ax=ax)
sns.despine(fig)
ax.set_ylim(0, y_max)
ax.set_xlabel('Sample mean')
ax.set_title('Sample size = 25')
ax.set_axisbelow(True) # grid lines behind data
ax.grid(True, linestyle=':', color='lightgray')

# part 4 - analyze the subsample means
print('\nRepeated subsample statistics')
print(f'E[mean] = {np.mean(sample_means)}')
print(f'E[std]  = {np.mean(sample_std)}')

# part 5 - repeat with k = 40
samples = np.stack(
    [subsample(sample, k=40) for i in range(1000)]
)
sample_means = np.mean(samples, axis=1)
sample_std   = np.std(samples, axis=1)

ax = axes[1]
sns.histplot(sample_means, bins=np.linspace(12, 18, 21), ax=ax)
sns.despine(fig)
ax.set_ylim(0, y_max)
ax.set_ylabel('')
ax.set_xlabel('Sample mean')
ax.set_title('Sample size = 40')
ax.set_axisbelow(True) # grid lines behind data
ax.grid(True, linestyle=':', color='lightgray')
fig.tight_layout()
fig.savefig('sample_mean_hists.png', bbox_inches='tight', dpi=400)

# part 6 - confidence intervals
print('\nSingle subsample confidence interval')
confidence_interval(sample[:25], confidence_level=0.95)
print()
