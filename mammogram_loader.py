import cv2
import sys
import random
from os import listdir
from os.path import isfile, join
import dtcwt

import numpy as np

# try:
#     import cupy as np
# except ImportError:
#     import numpy as np

PATH_CANCER = "dataset/mammogram/cancer";
PATH_NORMAL = "dataset/mammogram/normal";

PATH_CANCER_TEST = "dataset/mammogram-test/cancer";
PATH_NORMAL_TEST = "dataset/mammogram-test/normal";

FILE_TIPE = ".png"

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
    output_cancer = np.array([[1]])
    cancer_list = [f for f in listdir(path_cancer) if isfile(join(path_cancer, f))]
    for cancer_file in cancer_list:
        # print "cancer_file : ",cancer_file
        if (cancer_file.endswith(FILE_TIPE)):
            input_cancer = np.asarray(cv2.imread(path_cancer+"/"+cancer_file, cv2.IMREAD_GRAYSCALE))
            input_cancer = 1 - (input_cancer/255.0) #normalize value
            dataset_cancer.append(tuple((input_cancer, output_cancer)))
    random.shuffle(dataset_cancer)
    n = len(dataset_cancer)

    # training_data.extend(dataset_cancer[:int(0.9*n)])
    # dataset_cancer = dataset_cancer[0:int(n/10)]
    training_data.extend(dataset_cancer)

    # validation_data.extend(dataset_cancer[int(0.8*n):int(0.9*n)])

    # test_data.extend(dataset_cancer[int(0.9*n):])
    if env == 'test':
        test_data.extend(dataset_cancer[:30])
    else:
        test_data.extend(dataset_cancer[:int(n / 10)])

    # load data normal
    dataset_normal = list()
    output_normal = np.array([[0]])
    normal_list = [f for f in listdir(path_normal) if isfile(join(path_normal, f))]
    for normal_file in normal_list:
        # print "normal_file : ",normal_file
        if (normal_file.endswith(FILE_TIPE)):
            input_normal = np.asarray(cv2.imread(path_normal+"/"+normal_file, cv2.IMREAD_GRAYSCALE))
            input_normal = 1 - (input_normal / 255.0)  # normalize value
            dataset_normal.append(tuple((input_normal, output_normal)))
    random.shuffle(dataset_normal)
    n = len(dataset_normal)

    # training_data.extend(dataset_normal[:int(0.9 * n)])
    # dataset_normal = dataset_normal[0:int(n / 10)]
    training_data.extend(dataset_normal)

    # validation_data.extend(dataset_normal[int(0.8 * n):int(0.9 * n)])

    # test_data.extend(dataset_normal[int(0.9 * n):])
    if env == 'test':
        test_data.extend(dataset_normal[:30])
    else:
        test_data.extend(dataset_normal[:int(n / 10)])

    # return (training_data, validation_data, test_data)
    random.shuffle(training_data)  # randomize training dataset
    # random.shuffle(test_data)  # randomize training dataset
    return (training_data, test_data)

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
    output_cancer = np.array([[1+1j]])
    cancer_list = [f for f in listdir(path_cancer) if isfile(join(path_cancer, f))]
    for cancer_file in cancer_list:
        # print "cancer_file : ",cancer_file
        if (cancer_file.endswith(FILE_TIPE)):
            input_cancer = np.asarray(cv2.imread(path_cancer+"/"+cancer_file, cv2.IMREAD_GRAYSCALE))
            input_cancer_n = 1 - (input_cancer/255.0) #normalize value
            input_cancer_c = transform.forward(input_cancer_n, nlevels=3)
            for i in range(6):
                dataset_cancer.append(tuple((input_cancer_c.highpasses[0][:, :, i], output_cancer)))
    random.shuffle(dataset_cancer)

    n = len(dataset_cancer)
    print "dataset cancer : ", str(n)

    # validation_data.extend(dataset_cancer[int(0.8 * n):int(0.9 * n)])

    # test_data.extend(dataset_cancer[int(0.9 * n):])
    if env == 'test':
        training_data.extend(dataset_cancer)
        test_data.extend(dataset_cancer[:30])
    else:
        training_data.extend(dataset_cancer[:int(n / 6)])
        test_data.extend(dataset_cancer[:int(n/60)])

    # load data normal
    dataset_normal = list()
    output_normal = np.array([[0+0j]])
    normal_list = [f for f in listdir(path_normal) if isfile(join(path_normal, f))]
    for normal_file in normal_list:
        # print "normal_file : ",normal_file
        if (normal_file.endswith(FILE_TIPE)):
            input_normal = np.asarray(cv2.imread(path_normal+"/"+normal_file, cv2.IMREAD_GRAYSCALE))
            input_normal_n = 1 - (input_normal / 255.0)  # normalize value
            input_normal_c = transform.forward(input_normal_n, nlevels=3)
            for i in range(6):
                dataset_normal.append(tuple((input_normal_c.highpasses[0][:, :, i], output_normal)))
    random.shuffle(dataset_normal)

    n = len(dataset_normal)
    print "dataset normal : ", str(n)

    if env == 'test':
        training_data.extend(dataset_normal)
        test_data.extend(dataset_normal[:30])
    else:
        training_data.extend(dataset_normal[:int(n / 6)])
        test_data.extend(dataset_normal[:int(n/60)])

    print "dataset total : ", len(training_data)
    random.shuffle(training_data)  # randomize training dataset
    # random.shuffle(test_data)  # randomize training dataset
    return (training_data, test_data)


def load_data_dtcwt2(env):
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
    output_cancer = np.array([[1+1j]])
    cancer_list = [f for f in listdir(path_cancer) if isfile(join(path_cancer, f))]
    for cancer_file in cancer_list:
        # print "cancer_file : ",cancer_file
        if (cancer_file.endswith(FILE_TIPE)):
            input_cancer = np.asarray(cv2.imread(path_cancer+"/"+cancer_file, cv2.IMREAD_GRAYSCALE))
            input_cancer_n = 1 - (input_cancer/255.0) #normalize value
            input_cancer_c = transform.forward(input_cancer_n, nlevels=3)

            h, w, d = input_cancer_c.highpasses[0].shape
            input_cancer_append = np.zeros((d, h, w)) + 0j

            for i in range(6):
                input_cancer_append[i] = input_cancer_c.highpasses[0][:, :, i]
            dataset_cancer.append(tuple((input_cancer_append, output_cancer)))

    random.shuffle(dataset_cancer)

    n = len(dataset_cancer)
    print "dataset cancer : ", str(n)
    training_data.extend(dataset_cancer)

    # validation_data.extend(dataset_cancer[int(0.8 * n):int(0.9 * n)])

    # test_data.extend(dataset_cancer[int(0.9 * n):])
    if env == 'test':
        test_data.extend(dataset_cancer[:30])
    else:
        test_data.extend(dataset_cancer[:int(n/10)])

    # load data normal
    dataset_normal = list()
    output_normal = np.array([[0+0j]])
    normal_list = [f for f in listdir(path_normal) if isfile(join(path_normal, f))]
    for normal_file in normal_list:
        # print "normal_file : ",normal_file
        if (normal_file.endswith(FILE_TIPE)):
            input_normal = np.asarray(cv2.imread(path_normal+"/"+normal_file, cv2.IMREAD_GRAYSCALE))
            input_normal_n = 1 - (input_normal / 255.0)  # normalize value
            input_normal_c = transform.forward(input_normal_n, nlevels=3)
            h, w, d = input_normal_c.highpasses[0].shape
            input_normal_append = np.zeros((d, h, w)) + 0j

            for i in range(6):
                input_normal_append[i] = input_normal_c.highpasses[0][:, :, i]

            dataset_normal.append(tuple((input_normal_append, output_normal)))
    random.shuffle(dataset_normal)

    n = len(dataset_normal)
    print "dataset normal : ", str(n)
    training_data.extend(dataset_normal)

    if env == 'test':
        test_data.extend(dataset_normal[:30])
    else:
        test_data.extend(dataset_normal[:int(n/10)])

    print "dataset total : ", len(training_data)
    random.shuffle(training_data)  # randomize training dataset
    # random.shuffle(test_data)  # randomize training dataset
    return (training_data, test_data)

def load_data_dtcwt3(env):
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
    output_cancer = np.array([[1+1j]])
    cancer_list = [f for f in listdir(path_cancer) if isfile(join(path_cancer, f))]
    for cancer_file in cancer_list:
        # print "cancer_file : ",cancer_file
        if (cancer_file.endswith(FILE_TIPE)):
            input_cancer = np.asarray(cv2.imread(path_cancer+"/"+cancer_file, cv2.IMREAD_GRAYSCALE))
            input_cancer_n = 1 - (input_cancer/255.0) #normalize value
            input_cancer_c = transform.forward(input_cancer_n, nlevels=3)
            dataset_cancer.append(tuple((input_cancer_c.highpasses[0][:, :, 0], output_cancer)))
    random.shuffle(dataset_cancer)

    n = len(dataset_cancer)
    print "dataset cancer : ", str(n)

    # validation_data.extend(dataset_cancer[int(0.8 * n):int(0.9 * n)])

    # test_data.extend(dataset_cancer[int(0.9 * n):])
    if env == 'test':
        training_data.extend(dataset_cancer)
        test_data.extend(dataset_cancer[:30])
    else:
        training_data.extend(dataset_cancer[:int(n)])
        test_data.extend(dataset_cancer[:int(n/10)])

    # load data normal
    dataset_normal = list()
    output_normal = np.array([[0+0j]])
    normal_list = [f for f in listdir(path_normal) if isfile(join(path_normal, f))]
    for normal_file in normal_list:
        # print "normal_file : ",normal_file
        if (normal_file.endswith(FILE_TIPE)):
            input_normal = np.asarray(cv2.imread(path_normal+"/"+normal_file, cv2.IMREAD_GRAYSCALE))
            input_normal_n = 1 - (input_normal / 255.0)  # normalize value
            input_normal_c = transform.forward(input_normal_n, nlevels=3)
            dataset_normal.append(tuple((input_normal_c.highpasses[0][:, :, 0], output_normal)))
    random.shuffle(dataset_normal)

    n = len(dataset_normal)
    print "dataset normal : ", str(n)

    if env == 'test':
        training_data.extend(dataset_normal)
        test_data.extend(dataset_normal[:30])
    else:
        training_data.extend(dataset_normal[:int(n)])
        test_data.extend(dataset_normal[:int(n/10)])

    print "dataset total : ", len(training_data)
    random.shuffle(training_data)  # randomize training dataset
    # random.shuffle(test_data)  # randomize training dataset
    return (training_data, test_data)