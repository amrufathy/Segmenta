import datetime
import os
import random
import string

import numpy as np
from PIL import Image


class kmeans():
    def __init__(self, filepath, k=10, r=100, outdir=None):
        self.now = ''.join(c for c in str(datetime.datetime.today())
                           if c in string.digits)[:12]

        self.outdir = os.path.expanduser(outdir) if outdir else outdir
        self.basename = os.path.splitext(os.path.basename(filepath))[0]

        img = Image.open(filepath)
        # img.show()
        self.pixels = np.array(img)
        m, n, cols = self.pixels.shape

        idx_lst = [(j, k) for j in range(m) for k in range(n)]
        idx_arr = np.array(idx_lst).reshape((m, n, 2))
        # 2D array (m, n, r, g, b)
        self.features = np.concatenate((idx_arr, self.pixels), axis=2).ravel().reshape((m * n, 5))

        def calculate_centroids():
            centroids = random.sample(list(self.features), k)
            old_centroids = [np.empty_like(centroids[0]) for _ in range(len(centroids))]

            num_iter = 0
            while not np.allclose(old_centroids, centroids, atol=2) and num_iter < r:
                print(num_iter)
                print(old_centroids)
                print(centroids)
                print(np.isclose(old_centroids, centroids, atol=2))
                num_iter += 1
                old_centroids = centroids.copy()
                centroids = self.update_centroids(centroids)

            return centroids

        self.centroids = calculate_centroids()
        print('calculated centroids')
        self.generate_image()

    def update_centroids(self, centroids):
        d_centroids = {tuple(k): np.zeros([6], dtype=np.float32) for k in centroids}

        # iterate on each pixel in the picture
        for arr in self.features:
            d_centroids[self.nearest_centroid(arr, centroids)] += \
                np.concatenate([[1.0], arr])

        # calculate average of features
        return [feature_vector[1:] / feature_vector[0] for feature_vector in d_centroids.values()
                if feature_vector[0] > 0]  # drop empty clusters

    def nearest_centroid(self, pixel, centroids):
        def euclidean_distance(v1, v2):
            return np.linalg.norm(v1 - v2)

        distances = [(k, euclidean_distance(pixel[2:], k[2:])) for k in centroids]
        best_k, _ = min(distances, key=lambda t: t[1])

        return tuple(best_k)

    def generate_image(self):
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


image = '16052.jpg'

if __name__ == '__main__':
    kmeans('../' + image, k=11)
