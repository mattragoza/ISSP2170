import sys, os, string
import numpy as np
import pandas as pd
import sklearn.cluster
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_context('paper')
colors = np.array(sns.color_palette('muted'))
colors2 = np.array(sns.color_palette('dark'))


def k_cluster(X, n_clusters=3, init=None, random_state=None, verbose=True):
	'''
	Perform K-means clustering of data.

	Args:
		X: (N, 2) array of input data.
	Returns:
		Y: (N,) array of assigned cluster indices.
		means: (n_cluster, 2) array of cluster means.
		dist2: Total squared distance from data points
			to assigned cluster means.
	'''
	if verbose:
		print('Performing K-means clustering')
		print(f'n_clusters = {n_clusters}')
		print(f'init = {init}')
		print(f'random_state = {random_state}')

	model = sklearn.cluster.KMeans(
		n_clusters=n_clusters, init=init or 'k-means++',
		random_state=random_state
	)
	Y = model.fit_predict(X)

	means = model.cluster_centers_

	sizes = np.eye(n_clusters)[Y].sum(axis=0)

	dists = model.transform(X)
	dists = np.min(dists, axis=1)
	dist2 = (dists**2).sum()

	if verbose:
		print(f'Cluster means:\n{means}')
		print(f'Cluster sizes: {sizes}')
		print(f'Total squared distance: {dist2:.4f}')

	return Y, means, dist2


## load the data set

data_file = 'data/clustering_data.csv'

print(f'Loading data from {data_file}')

data = pd.read_csv(data_file)
print(data)

## perform k-means clustering

results = []

print('\nPART A')

result_a = k_cluster(data, n_clusters=3, random_state=0)
results.append(result_a)

print('\nPART B')

result_b = k_cluster(data, n_clusters=4, random_state=0)
results.append(result_b)

print('\nPART C')

result_c = k_cluster(data, n_clusters=4, init='random', random_state=1)
results.append(result_c)

print('\nPART D')

min_dist2 = np.inf
best_state = None
for i in range(30):
	random_state = np.random.randint(1e6)
	candid_d = k_cluster(data, n_clusters=4, init='random', random_state=random_state, verbose=False)
	dist2 = candid_d[-1]
	if dist2 < min_dist2:
		print(f'New best: {dist2:.4f}')
		min_dist2 = dist2
		best_state = random_state

result_d = k_cluster(data, n_clusters=4, init='random', random_state=best_state)
results.append(result_d)

## plot the clustering results

plot_file = 'k_means_clusters.png'

print(f'Plotting clusters to {plot_file}')

fig, axes = plt.subplots(2, 2, figsize=(6.5, 6.5))

titles = [
	'Part A',
	'Part B',
	'Part C',
	'Part D-E'
]

for i, ax in enumerate(axes.flatten()):
	Y, means, dist2 = results[i]
	ax.scatter(*data.values.T, c=colors[Y], s=10, alpha=0.5)
	for j in range(means.shape[0]):
		ax.scatter(*means[j],
			c=colors2[j:j+1], s=100, marker='+', label=f'Cluster {j+1}'
		)
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
