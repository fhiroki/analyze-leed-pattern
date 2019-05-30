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
BLOCK_SIZE = 71
n = 1


def detect_outer_circle(img):
    ret, img_thresh = cv2.threshold(img, 90, 90, cv2.THRESH_TOZERO)
    img_thresh = cv2.medianBlur(img_thresh, 9)
    img_thresh = cv2.adaptiveThreshold(img_thresh, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 101, 0)
    img_thresh = cv2.medianBlur(img_thresh, 15)
    outer_circle = cv2.HoughCircles(img_thresh, cv2.HOUGH_GRADIENT, dp=1, minDist=70,
                                    param1=140, param2=10, minRadius=440, maxRadius=460)

    x, y = None, None
    if outer_circle is not None:
        x, y, r = np.around(outer_circle[0][0])
        img_white = img.copy()
        img_white[:] = 255
        cv2.circle(img_white, (x, y), r, 0, -1)
        # cv2.circle(img, (x, y), 10, (255, 0, 0), -1)  # draw center of circle
        img = np.where(img_white == 255, 255, img)

    return img, [x, y]


def detect_blob(img, img_mask):
    cimg = cv2.cvtColor(img_mask, cv2.COLOR_GRAY2RGB)
    circles = cv2.HoughCircles(img_mask, cv2.HOUGH_GRADIENT, dp=1, minDist=90,
                               param1=100, param2=6, minRadius=5, maxRadius=20)

    circles = np.around(circles[0])
    for i in circles:
        cv2.circle(cimg, (i[0], i[1]), 10, (0, 0, 255), -1)

    return cimg, circles


def detect_base_blob(theoretical_d, image_paths, voltages, isfilename=False):
    x_list = np.array(0)
    sintheta_list = np.array(0)
    for i in range(len(image_paths)):
        if isfilename:
            vector = detect(os.path.join(DATA_DIR, image_paths[i]), theoretical_d)
        else:
            vector = detect(image_paths[i], theoretical_d)

        if vector is not None:
            x = cv2.magnitude(vector[:, 0], vector[:, 1])

            (freq, bins, _) = plt.hist(x, bins=100, range=(0, 500))

            bin_freqs = []
            for j in range(100):
                if freq[j]:
                    bin_freqs.append([bins[j], freq[j]])

            cluster = []
            prev_bin = 0
            start = 0
            for j in range(len(bin_freqs)):
                current_bin = bin_freqs[j][0]
                if current_bin > prev_bin + 20 or j == len(bin_freqs) - 1:
                    if j != 0:
                        end = current_bin if j == len(bin_freqs) - 1 else bin_freqs[j - 1][0]
                        x_extract = x[(x >= start) & (x <= end + 5)]
                        if len(x_extract) > 1:
                            cluster.append(x_extract)
                    start = current_bin
                prev_bin = current_bin

            x_list = np.append(x_list, np.median(cluster[0]))
            sintheta = n / theoretical_d * np.sqrt(150.4 / voltages[i])
            sintheta_list = np.append(sintheta_list, sintheta)

    plt.scatter(sintheta_list, x_list)
    plt.xlim([0, 0.6])
    plt.ylim([0, 500])

    x = sintheta_list / np.sqrt(1 - sintheta_list ** 2)
    r, intercept = np.polyfit(x, x_list, 1)
    plt.plot(x, np.poly1d([r, intercept])(x), label='r={}'.format(round(r, 2)))
    plt.legend()
    plt.show()

    return r


def detect(image_path, theoretical_d=0, dir_path=None, isplot=False):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if isplot:
        fig, axes = plt.subplots(2, 2)
        axes[0, 0].imshow(img)
        # axes[0, 0].set_title(filename)

    img_mask, center = detect_outer_circle(img)
    img_mask = cv2.adaptiveThreshold(img_mask, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, BLOCK_SIZE, 0)
    if center[0]:
        cv2.circle(img_mask, (center[0], center[1]), 80, 0, -1)

    if isplot:
        axes[0, 1].imshow(img_mask)
        axes[0, 1].set_title('adaptiveThreshold\nblock_size:{}'.format(BLOCK_SIZE))

    # モルフォロジー処理
    kernel_erode = np.ones((7, 7), np.uint8)
    kernel_dilate = np.ones((10, 10), np.uint8)
    img_mask = cv2.erode(img_mask, kernel_erode, iterations=1)
    img_mask = cv2.dilate(img_mask, kernel_dilate, iterations=1)

    if isplot:
        axes[1, 0].imshow(img_mask)
        axes[1, 0].set_title('morphology')

    vector = None
    try:
        cimg, circles = detect_blob(img, img_mask)
        vector = center - circles[:, :2]

        if isplot:
            axes[1, 1].imshow(cimg)
            axes[1, 1].set_title('detect circles')
    except:
        pass

    if ismultiple:
        plt.savefig(os.path.join(dir_path, filename))
    else:
        if isplot:
            plt.show()

    return vector


if __name__ == "__main__":
    if ismultiple:
        dir_name = datetime.now().strftime('%Y%m%d_%H%M')
        dir_path = os.path.join(OUT_DIR, dir_name)
        os.makedirs(dir_path, exist_ok=True)

        for i in tqdm(range(52)):
            filename = 'L16{}.tif'.format(450 + i)
            detect(filename, dir_path=dir_path)
    else:
        # detect(os.path.join(DATA_DIR, 'L16479.tif'), isplot=True)
        # detect_base_blob(2.504, ['L16480.tif'], [252.7], isfilename=True)
        detect_base_blob(2.504,
                         ['L16469.tif', 'L16470.tif', 'L16471.tif', 'L16472.tif', 'L16473.tif', 'L16474.tif',
                          'L16475.tif', 'L16476.tif', 'L16477.tif', 'L16478.tif', 'L16479.tif', 'L16480.tif',
                          'L16481.tif'],
                         [80.6, 94.7, 109.2, 122.9, 136.0, 150.9, 159.1, 179.2, 193.7, 215.3, 230.3, 252.7, 264.9],
                         isfilename=True)
