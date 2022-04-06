import sys, os, string
import numpy as np
import pandas as pd
import sklearn.cluster
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_context('paper')
colors = np.array(sns.color_palette('muted'))
colors2 = np.array(sns.color_palette('dark'))


def agg_cluster(X, n_clusters=3, linkage='complete', verbose=True):
	'''
	Perform agglomerative clustering of data.

	Args:
		X: (N, 2) array of input data.
	Returns:
		Y: (N,) array of assigned cluster indices.
	'''
	if verbose:
		print('Performing agglomerative clustering')
		print(f'n_clusters = {n_clusters}')
		print(f'linkage = {linkage}')

	model = sklearn.cluster.AgglomerativeClustering(
		n_clusters=n_clusters,
		linkage=linkage
	)
	Y = model.fit_predict(X)

	sizes = np.eye(n_clusters)[Y].sum(axis=0)

	if verbose:
		print(f'Cluster sizes: {sizes}')

	return Y


## load the data set

data_file = 'data/clustering_data.csv'

print(f'Loading data from {data_file}')

data = pd.read_csv(data_file)
print(data)

## perform agglomerative clustering

print('\nPART A')

Y_a = agg_cluster(data, n_clusters=3, linkage='complete')

print('\nPART B')

Y_b = agg_cluster(data, n_clusters=4, linkage='complete')

print('\nPART C')

Y_ward = agg_cluster(data, n_clusters=4, linkage='ward')
Y_avg = agg_cluster(data, n_clusters=4, linkage='average')
Y_min = agg_cluster(data, n_clusters=4, linkage='single')

## plot the clustering results

plot_file = 'agglom_clusters.png'

print(f'Plotting clusters to {plot_file}')

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
