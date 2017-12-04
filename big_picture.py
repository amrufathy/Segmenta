from lib.dataloader import load_test,test_paths
from lib.kmeans import KMeans
from lib.adjacency import knn, rbf
from lib.ncut import NCut, clustering
import numpy as np
import warnings
warnings.filterwarnings("ignore")

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
		image_name = image_path.split('/')[-1]
		assignment = kmeans.train(image_path)
		kmeans.generate_image().save('output/kmeans/'+image_name)
		assignments.append(assignment)
		print('Image #{} processed.\nAssignemnts: {}'.format(index+1,assignment))
	
	# ------------ Normalized Cut ------------ #

	print('\nRunning NCut on {} image(s)'.format(n_images))

	# Load test set for NCut
	test_set, test_gt = load_test('data')

	# Slice out first N images
	ncut_set = test_set[:n_images]

	# Apply NCut | 5-NN and K=5
	for index, image in enumerate(ncut_set):
		
		# Get image name
		image_name = test_paths[index].split('/')[-1]

		# Compute 5-NN matrix
		adjacency_matrix = knn(image,5)
		# Apply NCut on KNN
		assignment = NCut(adjacency_matrix,k=5)
		clustering(assignment,mode='get_image').savefig('output/ncut/knn/'+image_name)
		print('Image #{} processed using KNN.\n'.format(index+1))

		# Compute RBF matrix
		adjacency_matrix = rbf(image, gamma=10)
		# Apply NCut on RBF
		assignment = NCut(adjacency_matrix,k=5)
		clustering(assignment,mode='get_image').savefig('output/ncut/rbf/'+image_name)
		print('Image #{} processed using RBF.\n'.format(index+1))
