import os

import cv2
import numpy as np
import matplotlib.pyplot as plt

DATA_DIR = '../data/L16001-L17000/'
OUT_DIR = '../output/image/'
ismultiple = False


def detect_outer_circle(img):
    ret, img_thresh = cv2.threshold(img, 90, 90, cv2.THRESH_TOZERO)
    img_thresh = cv2.medianBlur(img_thresh, 9)
    img_thresh = cv2.adaptiveThreshold(img_thresh, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 101, 0)
    img_thresh = cv2.medianBlur(img_thresh, 15)
    outer_circle = cv2.HoughCircles(img_thresh, cv2.HOUGH_GRADIENT, dp=1, minDist=70,
                                    param1=140, param2=10, minRadius=440, maxRadius=460)

    if outer_circle is not None:
        x, y, r = np.around(outer_circle[0][0])
        mask = img.copy()
        cv2.circle(mask, (x, y), r, (0, 0, 0), -1)
        # cv2.circle(img, (x, y), 10, (255, 0, 0), -1)  # draw center of circle
        img -= mask

    return img


def detect_blob(img, img_mask):
    cimg = cv2.cvtColor(img_mask, cv2.COLOR_GRAY2RGB)
    circles = cv2.HoughCircles(img_mask, cv2.HOUGH_GRADIENT, dp=1, minDist=90,
                               param1=100, param2=6, minRadius=13, maxRadius=20)

    circles = np.around(circles[0])
    for i in circles:
        cv2.circle(cimg, (i[0], i[1]), 10, (0, 0, 255), -1)

    return cimg


def main(filename):
    # img = cv2.imread(os.path.join(DATA_DIR, filename), -1)  # read image as original 16 bit
    img = cv2.imread(os.path.join(DATA_DIR, filename), cv2.IMREAD_GRAYSCALE)
    fig, axes = plt.subplots(2, 2)
    axes[0, 0].imshow(img)
    axes[0, 0].set_title(filename)

    img = detect_outer_circle(img)
    ret, img_mask = cv2.threshold(img, 95, 95, cv2.THRESH_TOZERO)
    # img_mask = cv2.medianBlur(img_mask, 9)
    img_mask = cv2.adaptiveThreshold(img_mask, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 101, 0)
    axes[0, 1].imshow(img_mask)
    axes[0, 1].set_title('adaptiveThreshold')

    img_mask = cv2.medianBlur(img_mask, 15)
    axes[1, 0].imshow(img_mask)
    axes[1, 0].set_title('median blur')

    try:
        cimg = detect_blob(img, img_mask)
        # E = 51.1
        # r = 617.11
        # d = np.sqrt(150.4 / E) * np.sqrt(r ** 2 + x ** 2) / x

        axes[1, 1].imshow(cimg)
        axes[1, 1].set_title('detect circles')
    except:
        pass

    if ismultiple:
        plt.savefig(os.path.join(OUT_DIR, filename))
    else:
        plt.show()


if __name__ == "__main__":
    if ismultiple:
        for i in range(52):
            filename = 'L16{}.tif'.format(450 + i)
            main(filename)
    else:
        filename = 'L16501.tif'
        main(filename)
