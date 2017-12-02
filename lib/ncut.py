import numpy as np
import scipy as sp
import dataloader
import kmeans
import adjacency
from scipy.misc import imresize
def normalized_cut(adjacency_matrix, k):
	
	# Compute the degree matrix (delta)
	degree_matrix = np.zeros((adjacency_matrix.shape[0],adjacency_matrix.shape[0]))
	for row in range(adjacency_matrix.shape[0]):
		degree_matrix[row,row] = np.sum(adjacency_matrix[row,:])
	
	# Compute the laplacian matrix
	laplacian_matrix = degree_matrix - adjacency_matrix
	
	# Compute the eigenvectors and eigenvalues
	delta_inverse = np.linalg.pinv(degree_matrix)
	eigen_values, eigen_vectors = sp.sparse.linalg.eigs(np.dot(delta_inverse,laplacian_matrix))
	sorted_eigen_values = np.argsort(-eigen_values)

	# Fetch the K dominant eigenvectors
	dominant_indicies = sorted_eigen_values[:k]
	dominant_eigenvectors = eigen_vectors[dominant_indicies].T
	
	# Normalize dominant eigenvectors
	normalized_dominant_eigenvectors = np.zeros([dominant_eigenvectors.shape[0], k],dtype=complex)
	for value in range(dominant_eigenvectors.shape[0]):
		normalized_dominant_eigenvectors[value] = (1.0/np.sqrt(np.sum(np.square(dominant_eigenvectors[value,:]))))*(dominant_eigenvectors[value, :].T)
	
	# Apply K-means to normalized dominant eigenvectors
	# k_means = kmeans.KMeans(k=k,debug=False)
	# k_means.train(normalized_dominant_eigenvectors)
	# k_means.generate_image()
	
if __name__ == '__main__':

    # Load data
    data, data_gt = dataloader.load_test()

    # Fetch a single image
    image = data[0]

    # Apply normalized cut using KNN and RBF kernels
    rbf_adjacency = adjacency.rbf(image, gamma=1)
    normalized_cut(adjacency_matrix = rbf_adjacency, k = 5)
    knn_adjacency = adjacency.knn(image, n_neighbours=5)
    normalized_cut(adjacency_matrix = knn_adjacency, k = 5)
