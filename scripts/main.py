import os
from datetime import datetime

import cv2
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

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
        img_white = img.copy()
        img_white[:] = 255
        cv2.circle(img_white, (x, y), r, 0, -1)
        # cv2.circle(img, (x, y), 10, (255, 0, 0), -1)  # draw center of circle
        img = np.where(img_white == 255, 255, img)

    return img


def detect_blob(img, img_mask):
    cimg = cv2.cvtColor(img_mask, cv2.COLOR_GRAY2RGB)
    circles = cv2.HoughCircles(img_mask, cv2.HOUGH_GRADIENT, dp=1, minDist=90,
                               param1=100, param2=6, minRadius=5, maxRadius=20)

    circles = np.around(circles[0])
    for i in circles:
        cv2.circle(cimg, (i[0], i[1]), 10, (0, 0, 255), -1)

    return cimg


def main(filename, block_size=71, dir_path=None):
    # img = cv2.imread(os.path.join(DATA_DIR, filename), -1)  # read image as original 16 bit
    img = cv2.imread(os.path.join(DATA_DIR, filename), cv2.IMREAD_GRAYSCALE)
    fig, axes = plt.subplots(2, 2)
    axes[0, 0].imshow(img)
    axes[0, 0].set_title(filename)

    img_mask = detect_outer_circle(img)
    img_mask = cv2.adaptiveThreshold(img_mask, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, 0)
    axes[0, 1].imshow(img_mask)
    axes[0, 1].set_title('adaptiveThreshold\nblock_size:{}'.format(block_size))

    # 平滑化処理
    # img_mask = cv2.medianBlur(img_mask, ksize)

    # モルフォロジー処理
    kernel_erode = np.ones((7, 7), np.uint8)
    kernel_dilate = np.ones((10, 10), np.uint8)
    img_mask = cv2.erode(img_mask, kernel_erode, iterations=1)
    img_mask = cv2.dilate(img_mask, kernel_dilate, iterations=1)

    axes[1, 0].imshow(img_mask)
    axes[1, 0].set_title('morphology')

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
        plt.savefig(os.path.join(dir_path, filename))
    else:
        plt.show()


if __name__ == "__main__":
    if ismultiple:
        dir_name = datetime.now().strftime('%Y%m%d_%H%M')
        dir_path = os.path.join(OUT_DIR, dir_name)
        os.makedirs(dir_path, exist_ok=True)

        for i in tqdm(range(52)):
            filename = 'L16{}.tif'.format(450 + i)
            main(filename, dir_path=dir_path)

            # parameter search
            # for block_size in range(151, 301, 50):
            #     for ksize in range(19, 35, 2):
            #         main(filename, block_size, ksize, dir_path)
    else:
        filename = 'L16494.tif'
        main(filename)
