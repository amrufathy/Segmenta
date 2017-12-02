import os

from PIL import Image
from scipy.io import loadmat
from scipy.misc import imresize

def load_training(data_directory='../data', image_size=(30,30), image_depth=3):
    
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
        resized_image = imresize(image_file,image_size).reshape((image_size[0]*image_size[1],3))
        image_gt = loadmat(training_gt_directory + '/' + image_gt_file_name)
        training_set.append(resized_image)
        training_gt.append(image_gt)
    
    # Print stats
    print("Loaded training data succesfully. Set size =", training_count, "samples.")

    # Return sets
    return training_set, training_gt

def load_test(data_directory='../data', image_size=(30,30), image_depth=3):
    
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
        resized_image = imresize(image_file,image_size).reshape((image_size[0]*image_size[1],3))
        image_gt = loadmat(test_gt_directory + '/' + image_gt_file_name)
        test_set.append(resized_image)
        test_gt.append(image_gt)
    
    # Print stats
    print("Loaded test set succesfully. Set size =", test_count, "samples.")

    # Return sets
    return test_set, test_gt

def load_validation(data_directory='../data', image_size=(30,30), image_depth=3):
    
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
        resized_image = imresize(image_file,image_size).reshape((image_size[0]*image_size[1],3))
        image_gt = loadmat(validation_gt_directory + '/' + image_gt_file_name)
        validation_set.append(resized_image)
        validation_gt.append(image_gt)
    
    # Print stats
    print("Loaded validation set succesfully. Set size =", validation_count, "samples.")

    # Return sets
    return validation_set, validation_gt
