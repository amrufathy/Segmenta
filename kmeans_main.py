import os
import numpy as np
from lib.kmeans import KMeans
from lib.eval import conditional_entropy
from lib.data import load_test_files
import logging.handlers
from time import strftime

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
timestr = strftime("%Y-%m-%d")
handler = logging.handlers.TimedRotatingFileHandler('logs/' + timestr + ".log", when="H", interval=12)
formatter = logging.Formatter("%(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

DATA_DIR = os.path.join(
    os.getcwd(),
    'data',
    'images',
    'test'
)

abs_files_path = sorted(map(
    lambda file: os.path.join(DATA_DIR, file),
    os.listdir(DATA_DIR)
))

for k in [3, 5, 7, 9, 11]:
    kmeans = KMeans(k=k)
    print(k)

    for idx, img_file in enumerate(abs_files_path):
        assignments = kmeans.train(img_file)

        img_file = os.path.splitext(os.path.basename(img_file))[0]

        entropies = []

        for _idx, _val in enumerate(load_test_files(img_file)):
            segmentation, boundaries = _val
            c_e = conditional_entropy(assignments, segmentation)
            entropies.append(c_e)
            logger.info(
                'k = {}, img = {}, segmentation #{}, conditional_entropy = {:.3f}'.format(k, img_file, _idx + 1, c_e))

        logger.info('k = {}, img = {}, average_conditional_entropy = {:.3f}'.format(k, img_file, np.mean(entropies)))
        logger.info('=' * 100)

    logger.info('=' * 100)
    logger.info('=' * 100)