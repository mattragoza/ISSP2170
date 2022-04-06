import sys, os, string
import numpy as np
import pandas as pd
import sklearn.cluster
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_context('paper')
colors = np.array(sns.color_palette('muted'))


def fisher_score(x, y):
    assert np.all((y == 1) | (y == 0))
    x_pos = x[y == 1]
    x_neg = x[y == 0]
    return (
        (x_pos.mean() - x_neg.mean())**2 /
        (np.var(x_pos) + np.var(x_neg))
    )


def AUROC_score(x, y):
    assert np.all((y == 1) | (y == 0))
    return sklearn.metrics.roc_auc_score(y, x)


## load the data set

data_file = 'data/FeatureSelectionData.csv'

print(f'Loading data from {data_file}')

data = pd.read_csv(data_file)

print('\nPart A')

score = data[data.columns[:-1]].apply(fisher_score, y=data.Label)
print(score.sort_values(ascending=False).head(20))

print('\nPart B')

score = data[data.columns[:-1]].apply(AUROC_score, y=data.Label)
print(score.sort_values(ascending=False).head(20))

exit('HERE')

## plot the feature selection results

plot_file = 'feature_selection.png'

print(f'Plotting features to {plot_file}')

fig, axes = plt.subplots(1, 2, figsize=(6.5, 3.25))

results = [Y_a, Y_b]
titles = [
    'Part A',
    'Part B',
]
for i, ax in enumerate(axes.flatten()):
    Y = results[i]
    ax.scatter(*data.values.T, c=colors[Y], s=10, alpha=0.5)
    ax.legend(frameon=False, loc='center right')
    ax.set_title(titles[i], weight='bold')
    ax.set_axisbelow(True)
    ax.grid(True, color='lightgray', linestyle=':')
    ax.set_xlabel('Attribute 1')
    ax.set_ylabel('Attribute 2')

sns.despine(fig)
fig.tight_layout()
fig.savefig(plot_file, bbox_inches='tight', dpi=400)

## linkage comparison plot

plot_file = 'agglom_clusters2.png'

print(f'Plotting clusters to {plot_file}')

fig, axes = plt.subplots(2, 2, figsize=(6.5, 6.5))

results = [Y_b, Y_ward, Y_avg, Y_min]
titles = [
    'Max linkage',
    'Ward linkage',
    'Mean linkage',
    'Min linkage',
]
for i, ax in enumerate(axes.flatten()):
    Y = results[i]
    ax.scatter(*data.values.T, c=colors[Y], s=10, alpha=0.5)
    ax.legend(frameon=False, loc='center right')
    ax.set_title(titles[i], weight='bold')
    ax.set_axisbelow(True)
    ax.grid(True, color='lightgray', linestyle=':')
    ax.set_xlabel('Attribute 1')
    ax.set_ylabel('Attribute 2')

sns.despine(fig)
fig.tight_layout()
fig.savefig(plot_file, bbox_inches='tight', dpi=400)

print('Done')
