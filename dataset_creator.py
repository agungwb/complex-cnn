import cv2
import numpy as np
import sys

MASTER_FOLDER = "dataset/original/mias/master"
METADATA = "dataset/original/mias/master/metadata"
CANCER_FOLDER = "dataset/original/mias/processed/cancer"
NORMAL_FOLDER = "dataset/original/mias/processed/normal"

output_dimension = 36

def col_sum(z):
    return z.sum(axis=0)

def row_sum(z):
    return z.sum(axis=1)

def check_if_exists_zero(z):
    zero = False
    for data in z:
        if data == 0:
            zero = True
    return zero

def binarize(z):
    threshold = 150
    return np.where(z > threshold, 1, 0)

def translate_if_edge(coordinate, radius):
    y_from = coordinate[1] - radius
    y_to = coordinate[1] + radius
    x_from = coordinate[0] - radius
    x_to = coordinate[0] + radius

    if y_from < 0:
        diff = 0 - y_from
        y_from = 0
        y_to = y_to + diff

    if y_to > 1023:
        diff = y_to - 1023
        y_to = 1023
        y_from = y_from - diff

    if x_from < 0:
        diff = 0 - x_from
        x_from = 0
        x_to = x_to + diff

    if x_to > 1023:
        diff = x_to - 1023
        x_to = 1023
        x_from = x_from - diff

    return x_from, x_to, y_from, y_to


f = open(METADATA, "r")

for line in f:
    words = line.rstrip().split(" ")
    # print "words[0] : ",words[0]
    # print "words[1] : ",words[1]
    # print "words[2] : ",words[2]
    # print "---------"
    # print words[0]
    # filename = words[0]

    if (words[2] == "NORM"):
        print "---NORMAL---"


        img_name = words[0]
        img = cv2.imread('dataset/original/mias/master/' + img_name + '.pgm', cv2.IMREAD_GRAYSCALE)

        pictures_per_case = 39
        i = 1
        for i in range(pictures_per_case):
            up_limit = 923
            bottom_limit = 100

            black = True

            while black:
                random_x = int((np.random.rand() * (up_limit - bottom_limit)) + bottom_limit)
                random_y = int((np.random.rand() * (up_limit - bottom_limit)) + bottom_limit)
                random_coordinate = (random_x, random_y)

                up_radius = 100
                bottom_radius = 16
                random_radius = int((np.random.rand() * (up_radius - bottom_radius)) + bottom_radius)

                x_from, x_to, y_from, y_to = translate_if_edge(random_coordinate, random_radius)
                img_crop = img[y_from:y_to, x_from:x_to]
                img_binary = binarize(img_crop)

                horizontal_sum = row_sum(img_binary)
                horizontal_black = check_if_exists_zero(horizontal_sum)
                vertical_sum = col_sum(img_binary)
                vertical_black = check_if_exists_zero(vertical_sum)


                filename = words[0] + "_" + str(random_coordinate[0]) + "_" + str(random_coordinate[1]) + "_" + str(random_radius)

                print "filename : ",filename
                # print "img_binary : "
                # print img_binary
                # print "random_coordinate ; ", random_coordinate
                # print "random_radius ; ", random_radius
                # print "horizontal_sum : ", horizontal_sum
                # print "horizontal_black : ", horizontal_black
                # print "vertical_sum : ", vertical_sum
                # print "vertical_black : ", vertical_black
                black = horizontal_black or vertical_black

                if (not black):
                    img_resized = cv2.resize(img_crop, (output_dimension, output_dimension))
                    cv2.imwrite(NORMAL_FOLDER+"/"+filename+".png", img_resized)

    else:
        print "---CANCER---"
        continue

        ###START CANCER
        zooming_levels = [1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
        rotation_degrees = [0, 90, 180, 270]
        flip_methods = ['n', 'h', 'v']

        img_name = words[0]
        img = cv2.imread('dataset/original/mias/master/' + img_name + '.pgm', cv2.IMREAD_GRAYSCALE)

        for zooming_level in zooming_levels:
            for rotation_degree in rotation_degrees:
                for flip_method in flip_methods:

                    coordinate = (int(words[4]), 1023-int(words[5]))
                    radius = int(words[6])

                    #zooming level
                    radius = int(radius * zooming_level)
                    radius = radius if radius >= (output_dimension/2) else (output_dimension/2)

                    x_from, x_to, y_from, y_to = translate_if_edge(coordinate, radius)

                    img_crop = img[y_from:y_to, x_from:x_to]

                    if rotation_degree == 0:
                        img_rotated = img_crop
                    elif rotation_degree == 90:
                        img_rotated = np.rot90(img_crop)
                    elif rotation_degree == 180:
                        img_rotated = np.rot90(np.rot90(img_crop))
                    elif rotation_degree == 270:
                        img_rotated = np.rot90(np.rot90(np.rot90(img_crop)))


                    if flip_method == 'n':
                        img_flip = img_rotated
                    elif flip_method == 'h':
                        img_flip = np.fliplr(img_rotated)
                    elif flip_method == 'v':
                        img_flip = np.flipud(img_rotated)

                    filename = words[0]+'_'+str(int(zooming_level * 10))+'_'+str(rotation_degree)+'_'+flip_method

                    print "filename : ", filename
                    # print "coordinate : ", coordinate
                    # print "radius : ", radius
                    # print "then: ", (y_from, y_to, x_from, x_to)



                    if radius > (output_dimension/2):
                        img_resized = cv2.resize(img_flip,(output_dimension,output_dimension))
                        # img_resized = img_crop

                    cv2.imwrite(CANCER_FOLDER+"/"+filename+".png", img_resized)
            # sys.exit(0)
            ### END CANCER


# img = cv2.imread('dataset/original/mias/master/mdb063.pgm', cv2.IMREAD_GRAYSCALE)
# # coordinate = (546, 1023-463) #because origin from metadata is from left-bottom
# # radius = 33
#
# coordinate = (600, 550) #because origin from metadata is from left-bottom
# radius = 150
#
# # print "coordinate : ",coordinate
# # print "radius : ",radius
# #
# # print "img.shape : ", img.shape
# # print "img : ", img
# #
# # #in image x is column y is row, so x -> j, y -> i
# img_crop = img[coordinate[1]-radius:coordinate[1]+radius,coordinate[0]-radius:coordinate[0]+radius]
# # print "img_crop.shape : ", img_crop.shape
#
# test = np.array(img_crop)
#
# print "img_crop : ",img_crop
# col_sum = col_sum(img_crop)
# row_sum = row_sum(img_crop)
# col = check_if_exists_zero(col_sum)
# row = check_if_exists_zero(row_sum)
# print "col_sum : ", col_sum
# print "row_sum : ", row_sum
# print "col : ", col
# print "row : ", row
# print "resut : ", col or row
#
#
# # cv2.imshow("img", img)
# # cv2.waitKey(0)
# #
# cv2.imshow("img_crop", img_crop)
# cv2.waitKey(0)


# def checkIfBlack(sub_img):
#     a = [np.sum a[:,i] for i in sub_img]


