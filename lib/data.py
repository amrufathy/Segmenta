import os

from scipy.io import loadmat

ground_truth_path = os.path.join(
    os.getcwd(),
    'data',
    'groundTruth',
    'test'
)


def load_test_files(img):
    file_name = (os.path.splitext(img)[0] + '.mat') if img.endswith('.jpg') \
        else img + '.mat'

    abs_file_path = os.path.join(ground_truth_path, file_name)
    mat_file = loadmat(abs_file_path)

    for struct in mat_file['groundTruth'][0]:
        yield (
            struct[0]['Segmentation'][0],  # return as numpy arrays
            struct[0]['Boundaries'][0]
        )


def load_all_test_files():
    abs_files_path = list(map(
        lambda file: os.path.join(ground_truth_path, file),
        os.listdir(ground_truth_path)
    ))

    for img_file in abs_files_path:
        mat_file = loadmat(img_file)

        for struct in mat_file['groundTruth'][0]:
            yield (
                struct[0]['Segmentation'][0],  # return as numpy arrays
                struct[0]['Boundaries'][0]
            )


if __name__ == '__main__':
    for x, y in load_all_test_files():
        print(x, y)

    load_test_files('55075.jpg')
