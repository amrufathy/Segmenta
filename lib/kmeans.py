import random

import numpy as np
from PIL import Image


# noinspection SpellCheckingInspection
class KMeans:
    def __init__(self, k=10, r=100, debug=False):
        self.__k = k
        self.__r = r

        self.__debug = debug

    def train(self, filepath):
        """
        Extracts features from img ([locality,] color) and
            fits kmeans on them.
        """
        self.__extract_features(filepath)
        self.__fit()

        return self.__assign_clusters(self.__centroids)

    def __extract_features(self, filepath):
        img = Image.open(filepath)
        self.__pixels = np.array(img)
        m, n, cols = self.__pixels.shape
        if self.__debug: print('Read image: m = {}, n = {}, pixel: {}'.format(m, n, cols))

        if self.__debug: print('Extracting features...')
        idx_lst = [(j, k) for j in range(m) for k in range(n)]
        idx_arr = np.array(idx_lst).reshape((m, n, 2))
        # 2D array (x, y, r, g, b)
        self.__features = np.concatenate((idx_arr, self.__pixels), axis=2).ravel().reshape((m * n, 5))

    def __fit(self):
        if self.__debug: print('Fitting model on training data...')

        def calculate_centroids():
            centroids = random.sample(list(self.__features), self.__k)
            old_centroids = [np.empty_like(centroids[0]) for _ in range(len(centroids))]

            num_iter = 0
            while not np.allclose(old_centroids, centroids, atol=2) and num_iter < self.__r:
                if self.__debug: print(np.isclose(old_centroids, centroids, atol=2))

                num_iter += 1
                old_centroids = centroids.copy()
                centroids = self.__update_centroids(centroids)

            return centroids

        self.__centroids = calculate_centroids()
        if self.__debug: print('Calculated centroids...')

    def __update_centroids(self, centroids):
        assignments = self.__assign_clusters(centroids)
        new_centroids = []

        for k_idx in range(self.__k):
            cluster = self.__features[assignments == k_idx]

            if len(cluster) != 0:
                new_centroids.append(tuple(cluster.mean(axis=0)))

        return new_centroids

    def __assign_clusters(self, centroids):
        # Use RGB features only when calculating distances
        # Don't include locality
        # distances = np.abs(self.__features - np.array(centroids)[:, np.newaxis]).sum(axis=2)

        distances = np.sqrt(
            np.power(
                np.abs(self.__features[:, -3:] - np.array(centroids)[:, -3:][:, np.newaxis]), 2)
        ).sum(axis=2)

        return np.argmin(distances, axis=0)

    def generate_image(self):
        if self.__debug: print('Generating segmented image')
        new_pixels = np.empty_like(self.__pixels, dtype=np.uint8)

        d_clusters = self.__cluster()
        for pixel in d_clusters:
            x, y, r, g, b = pixel
            new_pixels[int(x), int(y)] = [int(r), int(g), int(b)]

        new_img = Image.fromarray(new_pixels)
        new_img.show()
        new_img.save('../segmented_' + image)

        return new_pixels

    def __cluster(self):
        pixels = []

        assignments = self.__assign_clusters(self.__centroids)

        for k_idx in range(self.__k):
            cluster = self.__features[assignments == k_idx]
            cluster_mean = cluster.mean(axis=0)

            for feature in cluster:
                pixels.append(
                    np.concatenate((
                        np.ravel(feature[:2]),
                        np.ravel(cluster_mean[-3:])
                    ))
                )

        return pixels


image = '55075.jpg'

if __name__ == '__main__':
    kmeans = KMeans(k=11, debug=False)
    kmeans.train('../' + image)
    kmeans.generate_image()
