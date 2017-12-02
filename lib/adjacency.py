import numpy as np
from sklearn.metrics import pairwise
from scipy.spatial import distance_matrix

def knn(data,n_neighbours=3):
	
	# Fetch sample count
	sample_count = data.shape[0]
	
	# Compute distances
	distances = distance_matrix(data,data)
	
	# Get the indexes of the neareast neighbours
	minimum_distances = np.argsort(distances, axis=1)
	knn_indexes = minimum_distances[:, 1:n_neighbours+1]
	
	# Create adjacency matrix, set the K nearest neighbours to 1
	adjacency_matrix = np.zeros([sample_count,sample_count])
	for row in range(sample_count):
		for column in range(n_neighbours):
			adjacency_matrix[row,knn_indexes[row,column]] = 1
	
	# Return KNN adjacency matrix
	return adjacency_matrix

def rbf(data, gamma=1):

	# Return RBF adjaceny matrix | Currently uses sklearn - Might be implemented from scratch later
	return pairwise.rbf_kernel(data,gamma=gamma)
