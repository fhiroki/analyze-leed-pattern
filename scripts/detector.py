import os
from datetime import datetime

import cv2
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

OUT_DIR = 'output/image/'
ismultiple = False
BLOCK_SIZE = 81
is111 = False  # TODO - implement GUI
n = 1


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
        img_white = img.copy()
        img_white[:] = 255
        cv2.circle(img_white, (x, y), r, 0, -1)
        # cv2.circle(img, (x, y), 10, (255, 0, 0), -1)  # draw center of circle
        img = np.where(img_white == 255, 255, img)

    return img, [x, y]


def detect_blob(img, img_mask):
    cimg = cv2.cvtColor(img_mask, cv2.COLOR_GRAY2RGB)
    circles = cv2.HoughCircles(img_mask, cv2.HOUGH_GRADIENT, dp=1, minDist=50,
                               param1=100, param2=6, minRadius=5, maxRadius=20)

    circles = np.around(circles[0])
    for i in circles:
        cv2.circle(cimg, (i[0], i[1]), 10, (0, 0, 255), -1)

    return cimg, circles


def detect_base_blob(theoretical_d, DATA_DIR, image_paths, voltages, is111=True, isfilename=False, isplot=False):
    xs = np.array(0)
    sinthetas = np.array(0)
    delta_bin = 10

    for i in range(len(image_paths)):
        if isfilename:
            vector = detect(os.path.join(DATA_DIR, image_paths[i]))
        else:
            vector = detect(image_paths[i])

        if vector is not None:
            x = cv2.magnitude(vector[:, 0], vector[:, 1])

            # clustering of x
            freq, bins = np.histogram(x, bins=100, range=(0, 500))
            bin_freqs = []
            for j in range(100):
                if freq[j]:
                    bin_freqs.append([bins[j], freq[j]])

            cluster = []
            prev_bin = 0
            start = 0
            for j in range(len(bin_freqs)):
                current_bin = bin_freqs[j][0]
                if current_bin > prev_bin + delta_bin or j == len(bin_freqs) - 1:
                    if j != 0:
                        end = current_bin if j == len(bin_freqs) - 1 else bin_freqs[j - 1][0]
                        x_ex = x[(x >= start) & (x <= end + delta_bin)]
                        if len(x_ex) > 1:
                            cluster.append(x_ex)
                    start = current_bin
                prev_bin = current_bin

            # TODO - use cluster >= 1
            if is111:
                xs = np.append(xs, np.median(cluster[0]))
                sintheta = n / theoretical_d * np.sqrt(150.4 / voltages[i])
                sinthetas = np.append(sinthetas, sintheta)
            else:
                if len(cluster) < 3:
                    continue
                is_show_inner_blob = np.median(cluster[1]) / np.median(cluster[0]) > 2.0
                if is_show_inner_blob:
                    xs = np.append(xs, np.median(cluster[0]))
                    sintheta = np.sqrt(150.4 / voltages[i]) / (theoretical_d * 2)
                    sinthetas = np.append(sinthetas, sintheta)
                    xs = np.append(xs, np.median(cluster[2]))
                    sintheta = np.sqrt(150.4 / voltages[i]) / (theoretical_d / 2**0.5)
                    sinthetas = np.append(sinthetas, sintheta)
                else:
                    xs = np.append(xs, np.median(cluster[0]))
                    sintheta = np.sqrt(150.4 / voltages[i]) / theoretical_d
                    sinthetas = np.append(sinthetas, sintheta)
                    xs = np.append(xs, np.median(cluster[1]))
                    sintheta = np.sqrt(150.4 / voltages[i]) / (theoretical_d / 2**0.5)
                    sinthetas = np.append(sinthetas, sintheta)

    x = sinthetas / np.sqrt(1 - sinthetas ** 2)
    r, intercept = np.polyfit(x, xs, 1)

    if isplot:
        plt.scatter(sinthetas, xs)
        plt.xlim([0, 0.6])
        plt.ylim([0, 500])

        plt.plot(x, np.poly1d([r, intercept])(x), label='r={}'.format(round(r, 2)))
        plt.legend()
        plt.show()

    return r


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


def detect(image_path, dir_path=None, isplot=False):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if isplot:
        fig, axes = plt.subplots(2, 2)
        axes[0, 0].imshow(img)
        axes[0, 0].set_title(image_path.split('/')[-1])

    img_mask, center = detect_outer_circle(img)
    img_mask = cv2.adaptiveThreshold(img_mask, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, BLOCK_SIZE, 0)
    if center[0]:
        cv2.circle(img_mask, (center[0], center[1]), 80, 0, -1)

    if isplot:
        axes[0, 1].imshow(img_mask)
        axes[0, 1].set_title('adaptiveThreshold')

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
        # detect(os.path.join(DATA_DIR, 'L16548.tif'), isplot=True)
        # detect(os.path.join(DATA_DIR, 'L16491.tif'), isplot=True)

        # Test of 6P/Au(110)
        Au_a = 4.07864
        theoretical_d = Au_a
        r = detect_base_blob(theoretical_d, 'data/6P_Au110/LEED_data',
                             ['L16543.tif', 'L16544.tif', 'L16545.tif', 'L16547.tif',
                                 'L16548.tif', 'L16549.tif', 'L16550.tif'],
                             [134.4, 123.1, 115.7, 83.8, 79.9, 68.8, 63.4],
                             is111=is111, isfilename=True, isplot=True)

        # Test of Coronene/Cu(111)
        Cu_a = 3.61496
        theoretical_d = (Cu_a*2**0.5/2)*3**0.5/2
        r = detect_base_blob(theoretical_d, 'data/Coronene_Cu111/base',
                             ['L4898.tif', 'L4899.tif', 'L4900.tif', 'L4901.tif', 'L4902.tif', 'L4903.tif'],
                             [98.9, 131.6, 141.1, 177.9, 231.6, 321.2],
                             isfilename=True, isplot=True)

        # Test of Coronene/Ag(111)
        Ag_a = 4.0862
        theoretical_d = (Ag_a*2**0.5/2)*3**0.5/2
        r = detect_base_blob(theoretical_d, 'data/Coronene_Ag111/image/base',
                             ['L16469.tif', 'L16470.tif', 'L16471.tif', 'L16472.tif', 'L16473.tif', 'L16474.tif',
                              'L16475.tif', 'L16476.tif', 'L16477.tif', 'L16478.tif', 'L16479.tif', 'L16480.tif',
                              'L16481.tif'],
                             [80.6, 94.7, 109.2, 122.9, 136.0, 150.9, 159.1,
                                 179.2, 193.7, 215.3, 230.3, 252.7, 264.9],
                             isfilename=True, isplot=True)

        # detect_mole_blob(617.11,
        #                  ['L16495.tif', 'L16496.tif', 'L16498.tif', 'L16499.tif', 'L16500.tif', 'L16501.tif'],
        #                  [15.7, 43.4, 23.9, 42.3, 45.3, 51.1],
        #                  isfilename=True)
