from collections import Counter

import numpy as np

from lib.data import load_segmentations
from lib.kmeans import KMeans


# noinspection PyShadowingNames
def f1_score(assignments, ground_truth):
    assignments, ground_truth = np.ravel(assignments), np.ravel(ground_truth)
    f_measures = []

    ground_truth_counter = Counter(ground_truth)

    for cluster in np.unique(assignments):
        # n_c,k samples from class c assigned to cluster k
        cluster_elements = ground_truth[np.where(assignments == cluster)]
        # count of n_c,k
        elements_counter = Counter(cluster_elements)

        max_vote = elements_counter.most_common(1)[0]

        purity = max_vote[1] / np.size(cluster_elements)
        recall = max_vote[1] / ground_truth_counter[max_vote[0]]

        f_measures.append(
            2 * purity * recall / (purity + recall)
        )

    return np.mean(f_measures)


# noinspection PyShadowingNames,SpellCheckingInspection
def conditional_entropy(assignments, ground_truth):
    # ref: http://scikit-learn.org/stable/modules/clustering.html#id15
    assignments, ground_truth = np.ravel(assignments), np.ravel(ground_truth)
    entropy = 0

    for cluster in np.unique(assignments):
        # n_c,k samples from class c assigned to cluster k
        cluster_elements = ground_truth[np.where(assignments == cluster)]
        # count of n_c,k
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

    for segmentation, boundaries in load_segmentations(image):
        print(conditional_entropy(assignments, segmentation))
