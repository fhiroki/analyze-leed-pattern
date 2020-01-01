import argparse
from datetime import datetime
import itertools
import os
import warnings

import cv2
import numpy as np
import matplotlib.pyplot as plt

from leed.detector import detect
from leed.utils import get_images_and_voltages
warnings.filterwarnings('ignore')


def setup_argument_parser(parser):
    """
    Set argument
    """
    parser.add_argument('--input-images-dir', help='input images directory', required=True)
    parser.add_argument('--input-voltages-path', help='input image, beam voltage csv file', required=True)
    parser.add_argument('--kind', help='base type', choices=['Au', 'Ag', 'Cu'], required=True)
    parser.add_argument('--surface', help='base surface', choices=['110', '111'], required=True)
    parser.add_argument('--isplot', help='draw a scatter plot of sinθ and X', action='store_true')
    parser.add_argument('--output-image-path', help='output plot image path')
    parser.add_argument('--manual-r', help='calculated r by myself')


def plot_scatter(xs, sinthetas, base_type, manual_r, rprime, intercept, output_image_path):
    plt.scatter(sinthetas, xs)
    plt.xlim([0, 0.6])
    plt.ylim([0, 500])
    plt.xlabel("sin?")
    plt.ylabel("X'")

    plt.title('{}({})'.format(base_type['kind'], base_type['surface']))

    if manual_r:
        label = 'r={}, manual_r={}'.format(round(rprime, 2), manual_r)
    else:
        label = 'r={}'.format(round(rprime, 2))

    plt.plot(xs, np.poly1d([rprime, intercept])(xs), label=label)
    plt.legend()

    if output_image_path:
        plt.savefig(output_image_path)
        print('save figure at', output_image_path)
    else:
        plt.show()


def fit_xs_and_sinthetas(xs, sinthetas):
    x = sinthetas / np.sqrt(1 - sinthetas ** 2)
    rprime, intercept = np.polyfit(x, xs, 1)

    # remove outlier
    outlier = np.abs(rprime*x+intercept - xs) > 50
    x = np.insert(x[~outlier], 0, 0)
    xs = np.insert(xs[~outlier], 0, 0)
    sinthetas = np.insert(sinthetas[~outlier], 0, 0)
    rprime, intercept = np.polyfit(x, xs, 1)

    return rprime, intercept


def calc_x_and_sintheta(xs, sinthetas, base_type, x_cluster, theta_cluster, voltage):
    theta_baseline = np.ones(2) * 100
    a = {'Cu': 3.61496, 'Ag': 4.0862, 'Au': 4.07864}

    if base_type['surface'] == '111':
        n = 1  # In the case of 111 surfaces, use only n = 1 because place spots on concentric circles.
        theoretical_d = a[base_type['kind']]*6**0.5/4
        sintheta = (n / theoretical_d) * np.sqrt(150.4 / voltage)
        return np.median(x_cluster[0]), sintheta

    elif base_type['surface'] == '110':
        # In the case of 110 surfaces, the surface arrangement is complicated, so it is considered separately.

        if base_type['kind'] == 'Au':
            if theta_baseline[0] == 100:
                theta_baseline[0] = min(theta_cluster[0])

            for i in range(len(theta_cluster)):
                error = np.abs(theta_baseline[0] - min(theta_cluster[i]))
                if error < 0.1:
                    x = np.median(x_cluster[i])
                    lamb = np.sqrt(150.4 / voltage)
                    n = (x / lamb) // 100 + 1
                    sintheta = n / (2 * a[base_type['kind']]) * lamb
                    if n > 2:
                        continue

                    return x, sintheta
        else:
            if theta_baseline[0] == 100:
                theta_baseline[0] = min(theta_cluster[0])
            if len(theta_cluster) > 1:
                if theta_baseline[1] == 100:
                    theta_baseline[1] = min(theta_cluster[1])

            for i in range(len(theta_cluster)):
                if i > 2:
                    break
                for j in range(2):
                    error = np.abs(theta_baseline[j] - min(theta_cluster[i]))
                    if error < 0.1:
                        n = 1 if j == 0 else 2**0.5
                        sintheta = n * np.sqrt(150.4 / voltage) / a[base_type['kind']]
                        return np.median(x_cluster[i]), sintheta


def clustering_theta(theta_cluster):
    valid_cluster = np.zeros(len(theta_cluster))
    pair_theta_cluster = []

    for i in range(len(theta_cluster)):
        for t1, t2 in itertools.combinations(theta_cluster[i], 2):
            error = np.pi - np.abs(t1 - t2)
            if np.abs(error) < 0.1:
                valid_cluster[i] = 1
                theta_cluster[i] = (t1, t2)
                pair_theta_cluster.append((t1, t2))

    return theta_cluster, valid_cluster, pair_theta_cluster


def clustering_x(x, theta):
    freq, bins = np.histogram(x, bins=100, range=(0, 500))
    bin_freqs = []
    delta_bin = 10

    for i in range(100):
        if freq[i]:
            bin_freqs.append([bins[i], freq[i]])

    x_cluster = []
    theta_cluster = []
    prev_bin = 0
    start = 0

    for i in range(len(bin_freqs)):
        current_bin = bin_freqs[i][0]
        if current_bin > prev_bin + delta_bin or i == len(bin_freqs) - 1:
            if i != 0:
                end = current_bin if i == len(bin_freqs) - 1 else bin_freqs[i - 1][0]
                x_range = (x >= start) & (x <= end + delta_bin)
                if len(x[x_range]) > 1:
                    x_cluster.append(x[x_range])
                    theta_cluster.append(theta[x_range])
            start = current_bin
        prev_bin = current_bin

    return x_cluster, theta_cluster


def clustering(vector, surface):
    x, theta = cv2.cartToPolar(vector[:, 0], vector[:, 1])

    # To remove outliers, put together x of the roughly same size
    x_cluster, theta_cluster = clustering_x(x, theta)

    theta_cluster, valid_cluster, pair_theta_cluster = clustering_theta(theta_cluster)

    x_cluster = [x_cluster[i] for i in range(len(x_cluster)) if valid_cluster[i]]
    if len(x_cluster) == 0:
        return None, None

    theta_cluster = [theta_cluster[i] for i in range(len(theta_cluster)) if valid_cluster[i]]

    return x_cluster, theta_cluster


def calc_rprime(input_images_dir, base_type, input_voltages_path, isplot=False, output_image_path=None, manual_r=None):
    image_paths, voltages = get_images_and_voltages(input_images_dir, input_voltages_path)

    xs = np.array([0])
    sinthetas = np.array([0])

    for i in range(len(image_paths)):
        vector = detect(os.path.join(input_images_dir, image_paths[i]))

        if vector is not None:
            x_cluster, theta_cluster = clustering(vector, base_type['surface'])
            if i == 0:
                return 0

            if x_cluster is None:
                continue

            x, sintheta = calc_x_and_sintheta(xs, sinthetas, base_type, x_cluster, theta_cluster, voltages[i])
            xs = np.append(xs, x)
            sinthetas = np.append(sinthetas, sintheta)

    rprime, intercept = fit_xs_and_sinthetas(xs, sinthetas)

    if isplot:
        plot_scatter(xs, sinthetas, base_type, manual_r, rprime, intercept, output_image_path)

    return rprime


def main(args):
    base_type = {'kind': args.kind, 'surface': args.surface}
    rprime = calc_rprime(args.input_images_dir, base_type, args.input_voltages_path,
                         isplot=args.isplot, output_image_path=args.output_image_path, manual_r=args.manual_r)
    print("r: {}".format(rprime))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    setup_argument_parser(parser)
    args = parser.parse_args()
    main(args)
