import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize as normalize
from scipy.sparse.csgraph import laplacian
from scipy.misc import imshow as show_image

def NCut(adjacency_matrix, k):
	
	# Compute the laplacian matrix
	laplacian_matrix = laplacian(adjacency_matrix,normed=False)

	# Compute the eigenvectors and eigenvalues
	eigen_values, eigen_vectors = np.linalg.eigh(laplacian_matrix)

	# Fetch the K least eigenvectors
	least_eigenvectors = eigen_vectors[:,:k]

	# Normalize eigenvectors
	normalized_eigenvectors = np.divide(least_eigenvectors,normalize(least_eigenvectors,norm='l1'))
	normalized_eigenvectors = np.nan_to_num(normalized_eigenvectors)
	
	# Apply K-means to normalized eigenvectors
	k_means = KMeans(n_clusters = k)
	clustered_assignments = k_means.fit_predict(normalized_eigenvectors)
	
	# Return assignments
	return clustered_assignments

def clustering(clustered_assignments, mode='show_matrix'):
	
	dimension = int(np.sqrt(clustered_assignments.shape[0]))
	plt.axis('off')
	plt.figure(frameon=False)
	if mode=='show_matrix':
		plt.matshow(clustered_assignments.reshape(((dimension,dimension))))
	elif mode=='show_image':
		show_image(clustered_assignments.reshape(((dimension,dimension))))
	elif mode=='get_image':
		return plt.matshow(clustered_assignments.reshape(((dimension,dimension)))).figure

if __name__ == '__main__':

    # Load data
    data, data_gt = dataloader.load_test()

    # Fetch a single image and its ground truth
    image = data[0] # Image
    image_gt = data_gt[0] # Ground truth

    # Apply normalized cut using KNN
    knn_adjacency = adjacency.knn(image, n_neighbors = 5)
    clustered_assignments = NCut(adjacency_matrix = knn_adjacency, k = 11)
    show_clustering(clustered_assignments)
    
    # Apply normalized cut using RBF
    #rbf_adjacency = adjacency.rbf(image, gamma = 1)
    #clustered_assignments = NCut(adjacency_matrix = rbf_adjacency, k = 11)
    #show_clustering(clustering)