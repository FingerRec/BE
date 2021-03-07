import cv2
import os


def mixup(im1, im2, prob):
    img1 = cv2.imread(im1)
    img2 = cv2.imread(im2)
    img = img1 * (1-prob) + img2 * prob
    return img


name = '6arrowswithin30seconds_shoot_bow_f_nm_np1_fr_med_1/'
video1 = "/data/jinhuama/DataSet/hmdb51/" + name
index = 15
new_dir = "../experiments/visualizations/mixup/"
prob = 0.3

if not os.path.exists(new_dir + name):
    os.mkdir(new_dir + name)
for image in os.listdir(video1):
    image1 = video1 + image
    image2 = video1 + "img_{:05d}.jpg".format(index)
    new_img = mixup(image1, image2, prob)
    cv2.imwrite(new_dir + name + image, new_img)