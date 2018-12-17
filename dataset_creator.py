import cv2

MASTER_FOLDER = "dataset/original/mias/master"
METADATA = "dataset/original/mias/master/metadata"
CANCER_FOLDER = "dataset/original/mias/processed/cancer"
NORMAL_FOLDER = "dataset/original/mias/processed/normal"

f = open(METADATA, "r")
for line in f:
    words = line.rstrip().split(" ")
    # print "words[0] : ",words[0]
    # print "words[1] : ",words[1]
    # print "words[2] : ",words[2]
    # print "---------"
    # print words[0]
    filename = words[0]
    if (words[2] == "NORM"):
        print "normal"
    else:
        print "cancer"
        coordinate = (int(words[4]), 1023-int(words[5]))
        radius = int(words[6])
        radius = int(radius * 1.1)
        radius = radius if radius >= 28 else 28

        img = cv2.imread('dataset/original/mias/master/'+filename+'.pgm', cv2.IMREAD_GRAYSCALE)
        img_crop = img[coordinate[1] - radius:coordinate[1] + radius, coordinate[0] - radius:coordinate[0] + radius]
        print "filename : ", filename
        print "coordinate : ", coordinate
        print "radius : ", radius
        if radius > 28:
            img_resized = cv2.resize(img_crop,(56,56))

        cv2.imwrite(CANCER_FOLDER+"/"+filename+".png", img_resized)


img = cv2.imread('dataset/original/mias/master/mdb063.pgm', cv2.IMREAD_GRAYSCALE)
coordinate = (546, 1023-463) #because origin from metadata is from left-bottom
radius = 33

# print "coordinate : ",coordinate
# print "radius : ",radius
#
# print "img.shape : ", img.shape
# print "img : ", img
#
# #in image x is column y is row, so x -> j, y -> i
# img_crop = img[coordinate[1]-radius:coordinate[1]+radius,coordinate[0]-radius:coordinate[0]+radius]
# print "img_crop.shape : ", img_crop.shape


# cv2.imshow("img", img)
# cv2.waitKey(0)
#
# cv2.imshow("img_crop", img_crop)
# cv2.waitKey(0)