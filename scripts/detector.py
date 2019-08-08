import os

import cv2
import itertools
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

a = {'Cu': 3.61496, 'Ag': 4.0862, 'Au': 4.07864}
BLOCK_SIZE = 121


def detect(image_path, dir_path=None, isplot=False, ismultiple=False):
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
    kernel_erode = np.ones((8, 8), np.uint8)
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
        plt.savefig(os.path.join(dir_path, image_path.split('/')[-1]))
    else:
        if isplot:
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


def detect_base_blob(DATA_DIR, base_type, voltages, image_paths=None,
                     isfilename=False, isplot=False, manual_r=None):
    xs = np.array([0])
    sinthetas = np.array([0])
    theta_baseline = np.ones(2) * 100
    delta_bin = 10

    if image_paths is None:
        image_paths = sorted(os.listdir(DATA_DIR))
        image_paths = [f for f in image_paths if f.endswith('tif')]

    # sort images and voltages by valtages
    image_paths = [x for _, x in sorted(zip(voltages, image_paths))]
    voltages = np.sort(voltages)

    for i in range(len(image_paths)):
        if isfilename:
            vector = detect(os.path.join(DATA_DIR, image_paths[i]))
            # vector = detect(os.path.join(DATA_DIR, image_paths[i]), isplot=True)
        else:
            vector = detect(image_paths[i])

        if vector is not None:
            x, theta = cv2.cartToPolar(vector[:, 0], vector[:, 1])

            # clustering of x
            freq, bins = np.histogram(x, bins=100, range=(0, 500))
            bin_freqs = []
            for j in range(100):
                if freq[j]:
                    bin_freqs.append([bins[j], freq[j]])

            cluster = []
            cluster_theta = []
            prev_bin = 0
            start = 0
            for j in range(len(bin_freqs)):
                current_bin = bin_freqs[j][0]
                if current_bin > prev_bin + delta_bin or j == len(bin_freqs) - 1:
                    if j != 0:
                        end = current_bin if j == len(bin_freqs) - 1 else bin_freqs[j - 1][0]
                        x_range = (x >= start) & (x <= end + delta_bin)
                        if len(x[x_range]) > 1:
                            cluster.append(x[x_range])
                            cluster_theta.append(theta[x_range])
                    start = current_bin
                prev_bin = current_bin

            valid_cluster = np.zeros(len(cluster))
            for j in range(len(cluster_theta)):
                for k in itertools.combinations(cluster_theta[j], 2):
                    error = np.pi - np.abs(k[0] - k[1])
                    if np.abs(error) < 0.1:
                        valid_cluster[j] = 1
                        cluster_theta[j] = k
            cluster = [cluster[j] for j in range(len(cluster)) if valid_cluster[j]]
            cluster_theta = [cluster_theta[j] for j in range(len(cluster_theta)) if valid_cluster[j]]
            if len(cluster) == 0:
                continue

            if base_type['surface'] == '111':
                theoretical_d = (a[base_type['kind']]/2**0.5)*3**0.5/2
                xs = np.append(xs, np.median(cluster[0]))
                sintheta = np.sqrt(150.4 / voltages[i]) / theoretical_d
                sinthetas = np.append(sinthetas, sintheta)
            elif base_type['surface'] == '110':
                if base_type['kind'] == 'Au':
                    if theta_baseline[0] == 100:
                        theta_baseline[0] = min(cluster_theta[0])

                    for j in range(len(cluster_theta)):
                        error = np.abs(theta_baseline[0] - min(cluster_theta[j]))
                        if error < 0.1:
                            x = np.median(cluster[j])
                            lamb = np.sqrt(150.4 / voltages[i])
                            n = (x / lamb) // 100 + 1
                            sintheta = n / (2 * a[base_type['kind']]) * lamb

                            xs = np.append(xs, x)
                            sinthetas = np.append(sinthetas, sintheta)
                else:
                    if theta_baseline[0] == 100:
                        theta_baseline[0] = min(cluster_theta[0])
                    if len(cluster_theta) > 1:
                        if theta_baseline[1] == 100:
                            theta_baseline[1] = min(cluster_theta[1])

                    for j in range(len(cluster_theta)):
                        if j > 2:
                            break
                        for k in range(2):
                            error = np.abs(theta_baseline[k] - min(cluster_theta[j]))
                            if error < 0.1:
                                xs = np.append(xs, np.median(cluster[j]))
                                n = 1 if k == 0 else 2**0.5
                                sintheta = n * np.sqrt(150.4 / voltages[i]) / a[base_type['kind']]
                                sinthetas = np.append(sinthetas, sintheta)

    x = sinthetas / np.sqrt(1 - sinthetas ** 2)
    r, intercept = np.polyfit(x, xs, 1)

    outlier = np.abs(r*x+intercept - xs) > 50
    x = np.insert(x[~outlier], 0, 0)
    xs = np.insert(xs[~outlier], 0, 0)
    sinthetas = np.insert(sinthetas[~outlier], 0, 0)
    r, intercept = np.polyfit(x, xs, 1)

    if isplot:
        plt.scatter(sinthetas, xs)
        plt.xlim([0, 0.6])
        plt.ylim([0, 500])

        plt.title('{}({})'.format(base_type['kind'], base_type['surface']))
        plt.plot(x, np.poly1d([r, intercept])(x), label='r={}, manual_r={}'.format(round(r, 2), manual_r))
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
