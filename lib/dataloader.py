import os

from PIL import Image
from scipy.io import loadmat
from scipy.misc import imresize

# Default dimenstions
default_dimensions = (70,70)
default_depth = 3

def load_training(data_directory='../data', image_size=default_dimensions, image_depth=default_depth):
    
    # Define lists
    training_set = list()
    training_gt = list()

    # Load training set and its ground truth
    training_directory = data_directory + '/images/train'
    training_gt_directory = data_directory + '/groundTruth/train'
    training_count = len(os.walk(training_directory).__next__()[2])
    training_images_names = os.listdir(training_directory)
    for image_name in training_images_names:
        image_gt_file_name = image_name.split('.')[0] + '.mat'
        image_file = Image.open(training_directory + '/' + image_name)
        resized_image = imresize(image_file,image_size).reshape((image_size[0]*image_size[1],image_depth))
        image_gt = loadmat(training_gt_directory + '/' + image_gt_file_name)
        training_set.append(resized_image)
        training_gt.append(image_gt)
    
    # Print stats
    print("Loaded training data succesfully. Set size =", training_count, "samples.")

    # Return sets
    return training_set, training_gt

def training_paths(data_directory='../data'):
    
    # Define lists
    training_set_paths = list()
    training_gt_paths = list()

    # Compute paths
    training_directory = data_directory + '/images/train'
    training_gt_directory = data_directory + '/groundTruth/train'
    training_count = len(os.walk(training_directory).__next__()[2])
    training_images_names = os.listdir(training_directory)
    for image_name in training_images_names:
        image_file_path = training_directory + '/' + image_name
        image_gt_file_path = training_directory + '/' + image_name.split('.')[0] + '.mat'
        training_set_paths.append(image_file_path)
        training_gt_paths.append(image_gt_file_path)

    # Return sets
    return training_set_paths, training_gt_paths

def load_test(data_directory='../data', image_size=default_dimensions, image_depth=default_depth):
    
    # Define lists
    test_set = list()
    test_gt = list()

    # Load test set and its ground truth
    test_directory = data_directory + '/images/test'
    test_gt_directory = data_directory + '/groundTruth/test'
    test_count = len(os.walk(test_directory).__next__()[2])
    test_images_names = os.listdir(test_directory)
    for image_name in test_images_names:
        image_gt_file_name = image_name.split('.')[0] + '.mat'
        image_file = Image.open(test_directory + '/' + image_name)
        resized_image = imresize(image_file,image_size).reshape((image_size[0]*image_size[1]),image_depth)
        image_gt = loadmat(test_gt_directory + '/' + image_gt_file_name)
        test_set.append(resized_image)
        test_gt.append(image_gt)
    
    # Print stats
    print("Loaded test set succesfully. Set size =", test_count, "samples.")

    # Return sets
    return test_set, test_gt

def test_paths(data_directory='../data'):
    
    # Define lists
    test_set_paths = list()
    test_gt_paths = list()

    # Compute paths
    test_directory = data_directory + '/images/test'
    test_gt_directory = data_directory + '/groundTruth/test'
    test_count = len(os.walk(test_directory).__next__()[2])
    test_images_names = os.listdir(test_directory)
    for image_name in test_images_names:
        image_file_path = test_directory + '/' + image_name
        image_gt_file_path = test_directory + '/' + image_name.split('.')[0] + '.mat'
        test_set_paths.append(image_file_path)
        test_gt_paths.append(image_gt_file_path)

    # Return sets
    return test_set_paths, test_gt_paths

def load_validation(data_directory='../data', image_size=default_dimensions, image_depth=default_depth):
    
    # Define lists
    validation_set = list()
    validation_gt = list()

    # Load validation set and its ground truth
    validation_directory = data_directory + '/images/val'
    validation_gt_directory = data_directory + '/groundTruth/val'
    validation_count = len(os.walk(validation_directory).__next__()()[2])
    validation_images_names = os.listdir(validation_directory)
    for image_name in validation_images_names:
        image_gt_file_name = image_name.split('.')[0] + '.mat'
        image_file = Image.open(validation_directory + '/' + image_name)
        resized_image = imresize(image_file,image_size).reshape((image_size[0]*image_size[1],image_depth))
        image_gt = loadmat(validation_gt_directory + '/' + image_gt_file_name)
        validation_set.append(resized_image)
        validation_gt.append(image_gt)
    
    # Print stats
    print("Loaded validation set succesfully. Set size =", validation_count, "samples.")

    # Return sets
    return validation_set, validation_gt

def validation_paths(data_directory='../data'):
    
    # Define lists
    validation_set_paths = list()
    validation_gt_paths = list()

    # Compute paths
    validation_directory = data_directory + '/images/val'
    validation_gt_directory = data_directory + '/groundTruth/val'
    validation_count = len(os.walk(validation_directory).__next__()[2])
    validation_images_names = os.listdir(validation_directory)
    for image_name in validation_images_names:
        image_file_path = validation_directory + '/' + image_name
        image_gt_file_path = validation_directory + '/' + image_name.split('.')[0] + '.mat'
        validation_set_paths.append(image_file_path)
        validation_gt_paths.append(image_gt_file_path)

    # Return sets
    return validation_set_paths, validation_gt_paths
