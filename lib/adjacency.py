from sklearn.metrics import pairwise
from sklearn.neighbors import kneighbors_graph

def knn(data,n_neighbors=5):

	# Return KNN dense graph
	return kneighbors_graph(data, n_neighbors).todense()

def rbf(data, gamma=1):

	# Return RBF adjaceny matrix | Currently uses sklearn - Might be implemented from scratch later
	return pairwise.rbf_kernel(data, None, gamma=gamma)
