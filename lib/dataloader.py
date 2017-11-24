import datetime
import os
import random
import string
import numpy as np
from scipy.io import loadmat
from PIL import Image

def load_data(data_directory='../data'):

    # Define lists
    training_set = list()
    training_gt = list()
    test_set = list()
    test_gt = list()
    validation_set = list()
    validation_gt = list()

    # Load training set and its ground truth
    training_directory = data_directory + '/images/train'
    training_gt_directory = data_directory + '/groundTruth/train'
    training_count = len(os.walk(training_directory).next()[2])
    training_images_names = os.listdir(training_directory)
    for image_name in training_images_names:
        image_gt_file_name = image_name.split('.')[0]+'.mat'
        image_file = Image.open(training_directory+'/'+image_name)
        image_gt = loadmat(training_gt_directory+'/'+image_gt_file_name)
        training_set.append(image_file)
        training_gt.append(image_gt)

    # Load test set and its ground truth
    test_directory = data_directory + '/images/test'
    test_gt_directory = data_directory + '/groundTruth/test'
    test_count = len(os.walk(test_directory).next()[2])
    test_images_names = os.listdir(test_directory)
    for image_name in test_images_names:
        image_gt_file_name = image_name.split('.')[0]+'.mat'
        image_file = Image.open(test_directory+'/'+image_name)
        image_gt = loadmat(test_gt_directory+'/'+image_gt_file_name)
        test_set.append(image_file)
        test_gt.append(image_gt)
    
    # Load validation set and its ground truth
    validation_directory = data_directory + '/images/val'
    validation_gt_directory = data_directory + '/groundTruth/val'
    validation_count = len(os.walk(validation_directory).next()[2])
    validation_images_names = os.listdir(validation_directory)
    for image_name in validation_images_names:
        image_gt_file_name = image_name.split('.')[0]+'.mat'
        image_file = Image.open(validation_directory+'/'+image_name)
        image_gt = loadmat(validation_gt_directory+'/'+image_gt_file_name)
        validation_set.append(image_file)
        validation_gt.append(image_gt)

    # Print stats
    print "Loaded the dataset succesfully.\n\nTraining set size = ",training_count,"samples.\nTest set size = ",\
    test_count,"samples.\nValidation set size = ",validation_count,"samples.\n\nData set size = ",\
    (training_count+test_count+validation_count),"samples.\n"

    # Return sets
    return training_set,training_gt,test_set,training_gt,validation_set,validation_gt