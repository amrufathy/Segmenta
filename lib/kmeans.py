import random

import numpy as np
from PIL import Image


class KMeans():
    def __init__(self, k=10, r=100, debug=False):
        self.k = k
        self.r = r

        self.pixels = None
        self.features = None
        self.debug = debug

    def fit(self, filepath):
        """
        Extracts features from image ([locality,] color) and
            fits kmeans on them.
        """
        img = Image.open(filepath)
        self.pixels = np.array(img)
        m, n, cols = self.pixels.shape
        if self.debug: print('Read image: m = {}, n = {}, pixel: {}'.format(m, n, cols))

        if self.debug: print('Extracting features...')
        idx_lst = [(j, k) for j in range(m) for k in range(n)]
        idx_arr = np.array(idx_lst).reshape((m, n, 2))
        # 2D array (m, n, r, g, b)
        self.features = np.concatenate((idx_arr, self.pixels), axis=2).ravel().reshape((m * n, 5))

        self._fit()

    def _fit(self):
        if self.debug: print('Fitting model on training data...')

        def calculate_centroids():
            centroids = random.sample(list(self.features), self.k)
            old_centroids = [np.empty_like(centroids[0]) for _ in range(len(centroids))]

            num_iter = 0
            while not np.allclose(old_centroids, centroids, atol=2) and num_iter < self.r:
                if self.debug: print(np.isclose(old_centroids, centroids, atol=2))

                num_iter += 1
                old_centroids = centroids.copy()
                centroids = self.update_centroids(centroids)

            return centroids

        self.centroids = calculate_centroids()
        if self.debug: print('Calculated centroids...')

    def update_centroids(self, centroids):
        d_centroids = {tuple(k): np.zeros([len(self.features[0]) + 1], dtype=np.float32) for k in centroids}

        # iterate on each pixel in the picture
        for arr in self.features:
            d_centroids[self.nearest_centroid(arr, centroids)] += \
                np.concatenate([[1.0], arr])

        # calculate average of features
        return [feature_vector[1:] / feature_vector[0] for feature_vector in d_centroids.values()
                if feature_vector[0] > 0]  # drop empty clusters

    @staticmethod
    def nearest_centroid(pixel, centroids):
        def distance(v1, v2, order=2):
            return np.linalg.norm(v1 - v2, ord=order)

        # To cluster on RGB only we pass the last 3 components to the distance functions
        # To include locality pass the whole vector
        distances = [(k, distance(pixel[-3:], k[-3:])) for k in centroids]
        best_k, _ = min(distances, key=lambda t: t[1])

        return tuple(best_k)

    def generate_image(self):
        if self.debug: print('Generating segmented image')
        new_pixels = np.empty_like(self.pixels, dtype=np.uint8)

        d_clusters = self.cluster()
        for centroid, pixels in d_clusters.items():
            for pixel in pixels:
                x, y, r, g, b = pixel
                new_pixels[int(x), int(y)] = [int(r), int(g), int(b)]

        new_img = Image.fromarray(new_pixels)
        new_img.show()

        new_img.save('../segmented_' + image)

    def cluster(self):
        d_clusters = {tuple(k): [] for k in self.centroids}

        for feature in self.features:
            nearest = self.nearest_centroid(feature, self.centroids)
            rep = tuple(np.concatenate((np.ravel(feature[:2]),
                                        np.ravel(nearest[2:]))))
            d_clusters[nearest].append(rep)

        return d_clusters


image = '55075.jpg'

if __name__ == '__main__':
    # kmeans('../' + image, k=11)
    kmeans = KMeans(k=11, debug=True)
    kmeans.fit('../' + image)
    kmeans.generate_image()
