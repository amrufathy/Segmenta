from __future__ import print_function
from sklearn.metrics import pairwise
import numpy as np
from scipy.spatial import distance_matrix
from scipy.misc import imread,imresize

import dataloader
import kmeans


def apply_knn(data,n_neighbours=3):
	
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
	return adjacency_matrix

def apply_rbf(data, gamma=1):
	return pairwise.rbf_kernel(data,gamma=gamma)

def normalized_cut(adjacency_matrix, k):
	
	# Compute the degree matrix (delta)
	degree_matrix = np.zeros((adjacency_matrix.shape[0],adjacency_matrix.shape[0]))
	for row in range(adjacency_matrix.shape[0]):
		degree_matrix[row,row] = np.sum(adjacency_matrix[row,:])
	
	# Compute the laplacian matrix
	laplacian_matrix = degree_matrix - adjacency_matrix
	
	# Compute the eigenvectors and eigenvalues
	delta_inverse = np.linalg.pinv(degree_matrix)
	eigen_values, eigen_vectors = np.linalg.eig(np.dot(delta_inverse,laplacian_matrix))
	sorted_eigen_values = eigen_values.argsort()
	
	# Fetch the K dominant eigenvectors
	dominant_indicies = sorted_eigen_values[:k]
	dominant_eigenvectors = eigen_vectors[dominant_indicies].T
	
	# Normalize dominant eigenvectors
	normalized_dominant_eigenvectors = np.zeros([dominant_eigenvectors.shape[0], k])
	for value in range(dominant_eigenvectors.shape[0]):
		normalized_dominant_eigenvectors[value] = (1/ np.sqrt(np.sum(np.square(dominant_eigenvectors[value,:]))))*(dominant_eigenvectors[value, :].T)
		
	# Apply K-means to normalized dominant eigenvectors
	#kmeans(normalized_dominant_eigenvectors,k)

if __name__ == '__main__':

    # Load data
    image = imresize(imread('../data/images/test/3063.jpg'),(30,30)).reshape((900,3))
    
    # Apply normalized cut using KNN and RBF kernels

    rbf_adjacency = apply_rbf(image,gamma=1)
    knn_adjacency = apply_knn(image,n_neighbours=5)
    normalized_cut(adjacency_matrix = rbf_adjacency, k = 5)
    normalized_cut(adjacency_matrix = knn_adjacency, k = 5)