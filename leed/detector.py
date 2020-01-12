import warnings

import cv2
import numpy as np
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

a = {'Cu': 3.61496, 'Ag': 4.0862, 'Au': 4.07864}
BLOCK_SIZE = 121


def set_image(axes, img, title):
    axes.imshow(img)
    axes.axis('off')
    axes.set_title(title)


def detect(input_image_path, isplot=False, output_image_path=None):
    img = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)
    # Detect center point.
    img_thresh, center = detect_outer_circle(img)
    img_thresh = cv2.adaptiveThreshold(img_thresh, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, BLOCK_SIZE, 0)
    if center[0]:
        cv2.circle(img_thresh, (center[0], center[1]), 80, 0, -1)

    # Morphological operating.
    img_morph = cv2.erode(img_thresh, np.ones((8, 8), np.uint8), iterations=1)
    img_morph = cv2.dilate(img_morph, np.ones((10, 10), np.uint8), iterations=1)

    # Remove connected areas.
    cimg = img_morph.copy()
    contours, hierarchy = cv2.findContours(cimg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        if area > 1000:
            cimg = cv2.drawContours(cimg, [cnt], -1, 0, BLOCK_SIZE - 1)

    vector = None
    try:
        cimg, circles = detect_blob(cimg)
        vector = center - circles[:, :2]

        if isplot:
            fig, axes = plt.subplots(2, 2, figsize=(10, 8))
            set_image(axes[0, 0], img, input_image_path.split('/')[-1])
            set_image(axes[0, 1], img_thresh, 'adaptiveThreshold')
            set_image(axes[1, 0], img_morph, 'morphology')
            cv2.circle(cimg, (center[0], center[1]), 10, (255, 0, 0), -1)

            set_image(axes[1, 1], cimg, 'detect circles')
            if output_image_path:
                print('save figure as {}'.format(output_image_path))
                plt.savefig(output_image_path)
            else:
                plt.show()
    except:
        pass

    return vector


def detect_outer_circle(img):
    img_thresh = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 101, 0)
    img_thresh = cv2.medianBlur(img_thresh, 15)
    outer_circle = cv2.HoughCircles(img_thresh, cv2.HOUGH_GRADIENT, dp=1, minDist=70,
                                    param1=140, param2=10, minRadius=440, maxRadius=460)

    x, y = None, None
    if outer_circle is not None:
        x, y, r = np.around(outer_circle[0][0])
        mask = np.ones(img.shape) * 255
        cv2.circle(mask, (x, y), r, 0, -1)
        img = np.where(mask == 255, 255, img)

    return img, [x, y]


def detect_blob(img):
    cimg = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, dp=1, minDist=30,
                               param1=140, param2=4, minRadius=3, maxRadius=25)

    circles = np.around(circles[0])
    for i in circles:
        cv2.circle(cimg, (i[0], i[1]), 10, (0, 0, 255), -1)

    return cimg, circles
