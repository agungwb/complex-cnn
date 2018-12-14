import cv2
import numpy as np
import sys
import random
from os import listdir
from os.path import isfile, join
import dtcwt

PATH_CANCER = "dataset/mammogram/cancer";
PATH_NORMAL = "dataset/mammogram/normal";

PATH_CANCER_TEST = "dataset/mammogram-test/cancer";
PATH_NORMAL_TEST = "dataset/mammogram-test/normal";

def load_data(env):
    training_data = list()
    validation_data = list()
    test_data = list()

    path_cancer = PATH_CANCER
    path_normal = PATH_NORMAL

    if env == 'test':
        path_cancer = PATH_CANCER_TEST
        path_normal = PATH_NORMAL_TEST

    #load data cancer
    dataset_cancer = list()
    output_cancer = np.array([[0],[1]])
    cancer_list = [f for f in listdir(path_cancer) if isfile(join(path_cancer, f))]
    for cancer_file in cancer_list:
        # print "cancer_file : ",cancer_file
        input_cancer = cv2.imread(path_cancer+"/"+cancer_file, cv2.IMREAD_GRAYSCALE)
        input_cancer = input_cancer/255.0 #normalize value
        dataset_cancer.append(tuple((input_cancer, output_cancer)))
    random.shuffle(dataset_cancer)
    n = len(dataset_cancer)
    training_data.extend(dataset_cancer[:int(0.8*n)])
    validation_data.extend(dataset_cancer[int(0.8*n):int(0.9*n)])
    test_data.extend(dataset_cancer[int(0.9*n):])

    # load data normal
    dataset_normal = list()
    output_normal = np.array([[1], [0]])
    normal_list = [f for f in listdir(path_normal) if isfile(join(path_normal, f))]
    for normal_file in normal_list:
        # print "normal_file : ",normal_file
        input_normal = cv2.imread(path_normal+"/"+normal_file, cv2.IMREAD_GRAYSCALE)
        input_normal = input_normal / 255.0  # normalize value
        dataset_normal.append(tuple((input_normal, output_normal)))
    random.shuffle(dataset_normal)
    n = len(dataset_normal)
    training_data.extend(dataset_normal[:int(0.8 * n)])
    validation_data.extend(dataset_normal[int(0.8 * n):int(0.9 * n)])
    test_data.extend(dataset_normal[int(0.9 * n):])

    return (training_data, validation_data, test_data)

def load_data_dtcwt(env):
    transform = dtcwt.Transform2d()
    training_data = list()
    validation_data = list()
    test_data = list()

    path_cancer = PATH_CANCER
    path_normal = PATH_NORMAL

    if env == 'test':
        path_cancer = PATH_CANCER_TEST
        path_normal = PATH_NORMAL_TEST

        #load data cancer
    dataset_cancer = list()
    output_cancer = np.array([[0],[1]])
    cancer_list = [f for f in listdir(path_cancer) if isfile(join(path_cancer, f))]
    for cancer_file in cancer_list:
        # print "cancer_file : ",cancer_file
        input_cancer = cv2.imread(path_cancer+"/"+cancer_file, cv2.IMREAD_GRAYSCALE)
        input_cancer_n = input_cancer/255.0 #normalize value
        input_cancer_c = transform.forward(input_cancer_n, nlevels=3)
        for i in range(6):
            dataset_cancer.append(tuple((input_cancer_c.highpasses[0][:, :, i], output_cancer)))
    random.shuffle(dataset_cancer)
    n = len(dataset_cancer)
    training_data.extend(dataset_cancer[:int(0.8 * n)])
    validation_data.extend(dataset_cancer[int(0.8 * n):int(0.9 * n)])
    test_data.extend(dataset_cancer[int(0.9 * n):])

    # load data normal
    dataset_normal = list()
    output_normal = np.array([[1], [0]])
    normal_list = [f for f in listdir(path_normal) if isfile(join(path_normal, f))]
    for normal_file in normal_list:
        # print "normal_file : ",normal_file
        input_normal = cv2.imread(path_normal+"/"+normal_file, cv2.IMREAD_GRAYSCALE)
        input_normal_n = input_normal / 255.0  # normalize value
        input_normal_c = transform.forward(input_normal_n, nlevels=3)
        for i in range(6):
            dataset_normal.append(tuple((input_normal_c.highpasses[0][:, :, i], output_normal)))
    random.shuffle(dataset_normal)
    n = len(dataset_normal)
    training_data.extend(dataset_normal[:int(0.8 * n)])
    validation_data.extend(dataset_normal[int(0.8 * n):int(0.9 * n)])
    test_data.extend(dataset_normal[int(0.9 * n):])

    return (training_data, validation_data, test_data)



