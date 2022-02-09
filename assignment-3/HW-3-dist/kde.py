import numpy as np
from scipy import stats

nax = np.newaxis


def pairwise_distance(x, y):
	'''
	Compute pairwise distances.

	Args:
		x: Length N vector.
		y: Length M vector.
	Returns:
		N x M distance matrix.
	'''
	return np.abs(x[:,nax] - y[nax,:])


def kde(x, D, k):
	'''
	Generic kernel density estimator.

	Args:
		x: Length N vector of query points.
		D: Length M vector of data points.
		k: Distance-based kernel function.
	Returns:
		Length N vector of density values.
	'''
	# compute pairwise distance matrix
	dist = pairwise_distance(x, D)

	# mean kernel density at query points
	return k(dist).mean(axis=1)


def parzen_window_pdf(x, D, h):
	'''
	Parzen window kernel density estimator.

	Args:
		x: Length N vector of query points.
		D: Length M vector of data points.
		h: Parzen window diameter.
	Returns:
		Length N vector of density values.
	'''
	k = lambda dist: (dist <= h/2) / h
	return kde(x, D, k)


def gaussian_kernel_pdf(x, D, h):
	'''
	Gaussian kernel density estimator.

	Args:
		x: Length N vector of query points.
		D: Length M vector of data points.
		h: Gaussian kernel standard deviation.
	Returns:
		Length N vector of density values.
	'''
	k = lambda dist: stats.norm.pdf(dist, scale=h)
	return kde(x, D, k)


def knn_estimate_pdf(x, D, k):
	'''
	K-nearest neighbors density estimator.

	Args:
		x: Length N vector of query points.
		D: Length M vector of data points.
		k: Number of nearest neighbors.
	Returns:
		Length N vector of density values.
	'''
	# compute pairwise distances
	dist = pairwise_distance(x, D)

	# select h for each query point as twice
	#   the distance to the kth nearest neighbor
	h = 2*np.sort(dist, axis=1)[:,k]

	return k/(D.shape[0]*h)
