import os

import cv2
import numpy as np
import matplotlib.pyplot as plt

DIR_PATH = '../data/L16001-L17000/'


def detecter(img):
    cimg = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, dp=1, minDist=90,
                               param1=100, param2=6, minRadius=13, maxRadius=20)
    outer_circle = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, dp=1, minDist=70,
                                    param1=140, param2=10, minRadius=440, maxRadius=460)

    circles = np.around(circles[0])
    for i in circles:
        cv2.circle(cimg, (i[0], i[1]), 10, (0, 0, 255), -1)

    center = np.around(outer_circle[0][0])
    cv2.circle(cimg, (center[0], center[1]), center[2], (0, 255, 0), 2)
    cv2.circle(cimg, (center[0], center[1]), 10, (255, 0, 0), -1)

    # for center in np.around(outer_circle[0]):
    #     cv2.circle(cimg, (center[0], center[1]), center[2], (0, 255, 0), 2)
    #     cv2.circle(cimg, (center[0], center[1]), 10, (255, 0, 0), -1)

    vector = center - circles
    x, theta = cv2.cartToPolar(vector[:, 0], vector[:, 1])
    theta = np.pi - theta

    return x, theta, cimg


def main():
    img = cv2.imread(os.path.join(DIR_PATH, 'L16501.tif'), cv2.IMREAD_GRAYSCALE)
    ax1 = plt.subplot(221)
    ax1.imshow(img)
    ax1.set_title('original')

    ret, img_mask = cv2.threshold(img, 100, 100, cv2.THRESH_TOZERO)
    img_mask = cv2.medianBlur(img_mask, 9)

    ax2 = plt.subplot(222)
    img_mask = cv2.adaptiveThreshold(img_mask, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 101, 0)
    ax2.imshow(img_mask)
    ax2.set_title('adaptiveThreshold')

    img_mask = cv2.medianBlur(img_mask, 15)
    ax3 = plt.subplot(223)
    ax3.imshow(img_mask)
    ax3.set_title('median blur')

    try:
        x, theta, cimg = detecter(img_mask)
    except:
        pass

    # E = 51.1
    # r = 617.11
    # d = np.sqrt(150.4 / E) * np.sqrt(r ** 2 + x ** 2) / x

    ax4 = plt.subplot(224)
    ax4.imshow(cimg)
    ax4.set_title('detect circles')
    plt.show()


if __name__ == "__main__":
    main()
