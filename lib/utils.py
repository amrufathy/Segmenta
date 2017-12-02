import os

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from lib.data import load_segmentations


def show_image_segmentations(image):
    image_path = os.getcwd() + '/data/images/test/' + image + '.jpg'
    img = Image.open(image_path)

    f1, axarr1 = plt.subplots(2, 3)  # segmentations
    f2, axarr2 = plt.subplots(2, 3)  # boundaries

    plt.suptitle('Visualization of image ' + image + ' with its segmentation')

    axarr1[0, 0].imshow(img)
    axarr1[0, 0].set_title('Image')

    axarr2[0, 0].imshow(img)
    axarr2[0, 0].set_title('Image')

    i = 0
    j = 1

    for segmentation, boundary in load_segmentations(image):
        k = np.unique(np.ravel(segmentation)).size
        axarr1[i, j].imshow(segmentation)
        axarr1[i, j].set_title('Seg k = {}'.format(k))

        axarr2[i, j].imshow(boundary, cmap='Greys')

        i = i + 1 if j == 2 else i
        j = 0 if j + 1 == 3 else j + 1

    f1.savefig('image_seg_viz.png')
    f2.savefig('image_bound_viz.png')
