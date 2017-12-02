import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.sparse.csgraph import laplacian
from scipy.misc import imshow as show_image

import dataloader
import kmeans
import adjacency

def normalized_cut(adjacency_matrix, k):
	
	# Compute the degree matrix (delta)
	laplacian_matrix = laplacian(adjacency_matrix)

	# Compute the eigenvectors and eigenvalues
	eigen_values, eigen_vectors = np.linalg.eigh(laplacian_matrix)

	# Fetch the K least eigenvectors
	least_eigenvectors = eigen_vectors.T[:,:k]

	# Normalize eigenvectors
	normalized_eigenvectors = np.divide(least_eigenvectors,np.linalg.norm(least_eigenvectors))
	
	# Apply K-means to normalized eigenvectors
	k_means = KMeans(n_clusters = k)
	clustered_assignments = k_means.fit_predict(normalized_eigenvectors)
	
	# Return assignments
	return clustered_assignments

def show_clustering(clustered_assignments, mode='matrix'):
	
	dimension = int(np.sqrt(clustered_assignments.shape[0]))
	
	if mode=='matrix':
		plt.matshow(clustered_assignments.reshape(((dimension,dimension))))
		plt.show()
	elif mode=='image':
		show_image(clustered_assignments.reshape(((dimension,dimension))))

if __name__ == '__main__':

    # Load data
    data, data_gt = dataloader.load_test()

    # Fetch a single image and its ground truth
    image = data[0] # Image

    # Apply normalized cut using KNN and RBF kernels
    
    knn_adjacency = adjacency.knn(image, n_neighbours = 5)
    clustered_assignments = normalized_cut(adjacency_matrix = knn_adjacency, k = 3)
    show_clustering(clustered_assignments)

    #rbf_adjacency = adjacency.rbf(image, gamma = 1)
    #clustered_assignments = normalized_cut(adjacency_matrix = rbf_adjacency, k = 11)
    #show_clustering(clustering)