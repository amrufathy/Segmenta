import os

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from lib.data import load_segmentations


def show_image_segmentations(image):
    image_path = os.getcwd() + '/data/images/test/' + image + '.jpg'
    img = Image.open(image_path)

    f, axarr = plt.subplots(2, 3)

    plt.suptitle('Visualization of image ' + image + ' with its segmentation')

    axarr[0, 0].imshow(img)
    axarr[0, 0].set_title('Image')

    i = 0
    j = 1

    for segmentation, boundary in load_segmentations(image):
        k = np.unique(np.ravel(segmentation)).size
        axarr[i, j].imshow(segmentation)
        axarr[i, j].set_title('Seg k = {}'.format(k))

        i = i + 1 if j == 2 else i
        j = 0 if j + 1 == 3 else j + 1

    f.savefig('image_viz.png')
