from PIL import Image
from resizeimage import resizeimage
from os import listdir
from os.path import isfile, join

PATH_CANCER = "dataset/hanacaraka/ga";
PATH_NORMAL = "dataset/hanacaraka/ra";

PATH_CANCER_RESIZE = "dataset/hanacaraka-resized/ga";
PATH_NORMAL_RESIZE = "dataset/hanacaraka-resized/ra";

FILE_TIPE = ".jpg"

cancer_list = [f for f in listdir(PATH_CANCER) if isfile(join(PATH_CANCER, f))]

for cancer_file in cancer_list:
    # print "cancer_file : ",cancer_file
    if (cancer_file.endswith(FILE_TIPE)):
        fd_img = open(PATH_CANCER+"/"+cancer_file, 'r')
        img = Image.open(fd_img)
        img = resizeimage.resize_height(img, 24)
        new_path = PATH_CANCER_RESIZE+"/"+cancer_file
        img.save(new_path, img.format)
        print "image {0} has been created".format(new_path)
        fd_img.close()


normal_list = [f for f in listdir(PATH_NORMAL) if isfile(join(PATH_NORMAL, f))]
for normal_file in normal_list:
    # print "cancer_file : ",cancer_file
    if (normal_file.endswith(FILE_TIPE)):
        fd_img = open(PATH_NORMAL+"/"+normal_file, 'r')
        img = Image.open(fd_img)
        img = resizeimage.resize_height(img, 24)
        new_path = PATH_NORMAL_RESIZE+"/"+normal_file
        img.save(new_path, img.format)
        print "image {0} has been created".format(new_path)
        fd_img.close()