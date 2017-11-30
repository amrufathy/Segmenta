from collections import Counter

import numpy as np

from lib.data import load_test_files
from lib.kmeans import KMeans


def f1_measure(assignments, ground_truth):
    raise NotImplementedError


def conditional_entropy(assignments, ground_truth):
    # ref: http://scikit-learn.org/stable/modules/clustering.html#id15
    entropy = 0

    for cluster in np.unique(assignments):
        # n_c,k number of samples from class c assigned to cluster k
        cluster_elements = ground_truth[np.where(assignments == cluster)]
        elements_counter = Counter(cluster_elements)

        cluster_entropy = 0
        for _count in elements_counter.keys():
            # coeff = n_c,k / n
            # probability = n_c,k / n_c
            coeff = elements_counter[_count] / np.size(assignments)
            probability = elements_counter[_count] / np.size(cluster_elements)

            cluster_entropy -= coeff * np.log2(probability)

        entropy += cluster_entropy

    return entropy


image = '55075.jpg'

if __name__ == '__main__':
    kmeans = KMeans(k=11, debug=False)
    assignments = kmeans.train('../' + image)

    for segmentation, boundaries in load_test_files(image):
        print(conditional_entropy(assignments, segmentation))
