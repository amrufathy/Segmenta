from __future__ import print_function

import dataloader


def normalized_cut(data=None, k=3):
    print("Normalized cut method called. To be implemented.")


if __name__ == '__main__':
    # Load data
    training_set, training_gt, test_set, training_gt, validation_set, validation_gt = dataloader.load_data()

    # Apply normalized cut
    normalized_cut()
