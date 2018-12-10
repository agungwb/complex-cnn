import cv2
import numpy as np
import sys
from os import listdir
from os.path import isfile, join
import dtcwt

PATH_CANCER = "dataset/mammogram/cancer";
PATH_NORMAL = "dataset/mammogram/normal";

def load_data():
    dataset = list()

    #load data cancer
    output_cancer = np.array([[0],[1]])
    cancer_list = [f for f in listdir(PATH_CANCER) if isfile(join(PATH_CANCER, f))]
    for cancer_file in cancer_list:
        print "cancer_file : ",cancer_file
        input_cancer = cv2.imread(PATH_CANCER+"/"+cancer_file, cv2.IMREAD_GRAYSCALE)
        input_cancer = input_cancer/255.0 #normalize value
        dataset.append(tuple((input_cancer, output_cancer)))

    # load data normal
    output_normal = np.array([[1], [0]])
    normal_list = [f for f in listdir(PATH_NORMAL) if isfile(join(PATH_NORMAL, f))]
    for normal_file in normal_list:
        print "normal_file : ",normal_file
        input_normal = cv2.imread(PATH_NORMAL+"/"+normal_file, cv2.IMREAD_GRAYSCALE)
        input_normal = input_normal / 255.0  # normalize value
        dataset.append(tuple((input_normal, output_normal)))

    training_data = dataset
    validation_data = dataset
    test_data = dataset

    return (training_data, validation_data, test_data)

def load_data_dtcwt():
    transform = dtcwt.Transform2d()
    dataset = list()

    #load data cancer
    output_cancer = np.array([[0],[1]])
    cancer_list = [f for f in listdir(PATH_CANCER) if isfile(join(PATH_CANCER, f))]
    for cancer_file in cancer_list:
        # print "cancer_file : ",cancer_file
        input_cancer = cv2.imread(PATH_CANCER+"/"+cancer_file, cv2.IMREAD_GRAYSCALE)
        input_cancer_n = input_cancer/255.0 #normalize value
        input_cancer_c = transform.forward(input_cancer_n, nlevels=2)
        for i in range(6):
            dataset.append(tuple((input_cancer_c.highpasses[0][:, :, i], output_cancer)))

    # load data normal
    output_normal = np.array([[1], [0]])
    normal_list = [f for f in listdir(PATH_NORMAL) if isfile(join(PATH_NORMAL, f))]
    for normal_file in normal_list:
        # print "normal_file : ",normal_file
        input_normal = cv2.imread(PATH_NORMAL+"/"+normal_file, cv2.IMREAD_GRAYSCALE)
        input_normal_n = input_normal / 255.0  # normalize value
        input_normal_c = transform.forward(input_normal_n, nlevels=2)
        for i in range(6):
            dataset.append(tuple((input_normal_c.highpasses[0][:, :, i], output_normal)))

    training_data = dataset
    validation_data = dataset
    test_data = dataset

    return (training_data, validation_data, test_data)



