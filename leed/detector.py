import os
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
    img_thresh, center = detect_outer_circle(img)
    img_thresh = cv2.adaptiveThreshold(img_thresh, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, BLOCK_SIZE, 0)
    if center[0]:
        cv2.circle(img_thresh, (center[0], center[1]), 80, 0, -1)

    # モルフォロジー処理
    kernel_erode = np.ones((8, 8), np.uint8)
    kernel_dilate = np.ones((10, 10), np.uint8)
    img_mask = cv2.erode(img_thresh, kernel_erode, iterations=1)
    img_mask = cv2.dilate(img_mask, kernel_dilate, iterations=1)

    vector = None
    try:
        cimg, circles = detect_blob(img, img_mask)
        vector = center - circles[:, :2]
    except:
        pass

    if isplot:
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        set_image(axes[0, 0], img, input_image_path.split('/')[-1])
        set_image(axes[0, 1], img_thresh, 'adaptiveThreshold')
        set_image(axes[1, 0], img_mask, 'morphology')
        set_image(axes[1, 1], cimg, 'detect circles')
        if output_image_path:
            print('save figure as {}'.format(output_image_path))
            plt.savefig(output_image_path)
        else:
            plt.show()

    return vector


def detect_outer_circle(img):
    ret, img_thresh = cv2.threshold(img, 50, 50, cv2.THRESH_TOZERO)
    img_thresh = cv2.medianBlur(img_thresh, 9)
    img_thresh = cv2.adaptiveThreshold(img_thresh, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 101, 0)
    img_thresh = cv2.medianBlur(img_thresh, 15)
    outer_circle = cv2.HoughCircles(img_thresh, cv2.HOUGH_GRADIENT, dp=1, minDist=70,
                                    param1=140, param2=10, minRadius=440, maxRadius=460)

    x, y = None, None
    if outer_circle is not None:
        x, y, r = np.around(outer_circle[0][0])
        mask = np.ones(img.shape) * 255
        cv2.circle(mask, (x, y), r, 0, -1)
        # cv2.circle(img, (x, y), 10, (255, 0, 0), -1)  # draw center of circle
        img = np.where(mask == 255, 255, img)

    return img, [x, y]


def detect_blob(img, img_mask):
    cimg = cv2.cvtColor(img_mask, cv2.COLOR_GRAY2RGB)
    circles = cv2.HoughCircles(img_mask, cv2.HOUGH_GRADIENT, dp=1, minDist=50,
                               param1=100, param2=6, minRadius=5, maxRadius=20)

    circles = np.around(circles[0])
    for i in circles:
        cv2.circle(cimg, (i[0], i[1]), 10, (0, 0, 255), -1)

    return cimg, circles


def detect_mole_blob(r, DATA_DIR, image_paths, voltages, isfilename=False):
    d_invs = np.array([])
    thetas = np.array([])
    for i in range(len(image_paths)):
        if isfilename:
            vector = detect(os.path.join(DATA_DIR, image_paths[i]))
        else:
            vector = detect(image_paths[i])

        if vector is not None:
            x, theta = cv2.cartToPolar(vector[:, 0], vector[:, 1])
            theta = np.pi - theta
            d = np.sqrt(150.4 / voltages[i]) * np.sqrt(x ** 2 + r ** 2) / x
            d_inv = 1 / d

            d_invs = np.append(d_invs, d_inv)
            thetas = np.append(thetas, theta)

    d_invs = d_invs.flatten()
    thetas = thetas.flatten()

    # adjust angle
    freq, bins = np.histogram(thetas, bins=100, range=(-np.pi, np.pi))
    active_bins = [bins[i] for i in range(len(bins) - 1) if freq[i]]
    delta = np.pi / 50

    # parameter search
    num = 0
    n_bin = len(active_bins)
    fig = plt.figure(figsize=(24, 12))
    for a in range(1, 5):
        for b in range(1, 3):
            num += 1
            delta_bin = delta * a
            start, prev_bin = -np.pi, -np.pi
            d_invs_adj = d_invs.copy()
            thetas_adj = np.zeros(len(thetas))

            for i in range(n_bin):
                current_bin = active_bins[i]
                if current_bin > prev_bin + delta_bin or i == n_bin - 1:
                    if i != 0:
                        end = current_bin if i == n_bin - 1 else active_bins[i - 1]
                        theta_idx = np.where((thetas >= start) & (thetas < end + delta_bin))
                        theta_ex = thetas[theta_idx]
                        if len(theta_ex) > 1:
                            thetas_adj[theta_idx] = np.median(theta_ex)
                        else:
                            thetas_adj[theta_idx] = 100
                            d_invs_adj[theta_idx] = 0
                    start = current_bin
                prev_bin = current_bin

            # remove outlier
            delta_theta = delta * b
            for i in range(len(thetas_adj)):
                theta = thetas_adj[i] + np.pi if thetas_adj[i] < 0 else thetas_adj[i] - np.pi
                accept_range = ((thetas_adj >= theta - delta_theta) & (thetas_adj <= theta + delta_theta))
                if len(thetas_adj[accept_range]) < 1:
                    thetas_adj[i] = 100
                    d_invs_adj[i] = 0

            ax = fig.add_subplot(2, 4, num, projection='polar')
            ax.plot(thetas_adj, d_invs_adj, 'o')
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_title('a={},b={}'.format(a, b))
    plt.show()
