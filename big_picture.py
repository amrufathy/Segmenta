from lib.dataloader import load_test,test_paths
from lib.kmeans import KMeans
from lib.adjacency import knn
from lib.ncut import NCut


if __name__ == '__main__':

	# Number of images to use
	n_images = 5

	# ------------ K-means ------------ #

	print('Running K-means on {} image(s)'.format(n_images))

	# Fetch paths for K-means
	test_paths, test_gt_paths = test_paths('data')

	# Slice out first N paths
	kmeans_set_paths = test_paths[:n_images]

	# Apply K-means | K=5
	kmeans = KMeans(k=5)
	assignments = []
	for index, image_path in enumerate(kmeans_set_paths):
		assignment = kmeans.train(image_path)
		assignments.append(assignment)
		print('Image #{} processed.\nAssignemnts: {}'.format(index+1,assignment))
	
	# ------------ Normalized Cut ------------ #

	print('\nRunning NCut on {} image(s)'.format(n_images))

	# Load test set for NCut
	test_set, test_gt = load_test('data')

	# Slice out first N images
	ncut_set = test_set[:n_images]

	# Apply NCut | 5-NN and K=5
	assignments = []
	for index, image in enumerate(ncut_set):
		
		# Compute 5-NN matrix
		adjacency_matrix = knn(image, n_neighbours=5)
		
		# Apply NCut
		assignment = NCut(adjacency_matrix,k=5)
		assignments.append(assignment)
		print('Image #{} processed.\nAssignemnts: {}'.format(index+1,assignment))