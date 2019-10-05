import argparse
from datetime import datetime
import itertools
import os
import warnings

import cv2
import numpy as np
import matplotlib.pyplot as plt

from leed.detector import detect
warnings.filterwarnings('ignore')


def setup_argument_parser(parser):
    """
    Set argument
    """
    parser.add_argument('--input-dir', help='input images directory', required=True)
    parser.add_argument('--output-dir', help='output image directory')
    parser.add_argument('--kind', help='base type', choices=['Au', 'Ag', 'Cu'], required=True)
    parser.add_argument('--surface', help='base surface', choices=['110', '111'], required=True)
    parser.add_argument('--voltages', help='beam voltages (ex. 1.0,2.0,3.0)', required=True)
    parser.add_argument('--isplot', help='draw a scatter plot of sinθ and X', action='store_true')
    parser.add_argument('--manual-r', help='calculated r by myself')


def detect_base_blob(input_dir, base_type, voltages, image_paths=None,
                     isfilename=True, isplot=False, output_dir=None, manual_r=None):
    xs = np.array([0])
    sinthetas = np.array([0])
    theta_baseline = np.ones(2) * 100
    delta_bin = 10
    a = {'Cu': 3.61496, 'Ag': 4.0862, 'Au': 4.07864}

    if image_paths is None:
        image_paths = sorted(os.listdir(input_dir))
        image_paths = [f for f in image_paths if f.endswith('tif')]

    # sort images and voltages by valtages
    image_paths = [x for _, x in sorted(zip(voltages, image_paths))]
    voltages = np.sort(voltages)

    for i in range(len(image_paths)):
        if isfilename:
            vector = detect(os.path.join(input_dir, image_paths[i]))
            # vector = detect(os.path.join(input_dir, image_paths[i]), isplot=True)
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
                            if n > 2:
                                continue

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
        plt.xlabel("sinθ")
        plt.ylabel("X'")

        plt.title('{}({})'.format(base_type['kind'], base_type['surface']))
        plt.plot(x, np.poly1d([r, intercept])(x), label='r={}, manual_r={}'.format(round(r, 2), manual_r))
        plt.legend()
        filename = 'r_' + datetime.now().strftime('%Y%m%d_%H%M%S') + '.png'
        plt.savefig(os.path.join(output_dir, filename))
        print('save figure as {}'.format(os.path.join(output_dir, filename)))

    return r


def main(args):
    base_type = {'kind': args.kind, 'surface': args.surface}
    voltages = [float(voltage) for voltage in args.voltages.split(',')]
    r = detect_base_blob(args.input_dir, base_type, voltages,
                         isplot=args.isplot, output_dir=args.output_dir, manual_r=args.manual_r)
    print("r: {}".format(r))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    setup_argument_parser(parser)
    args = parser.parse_args()
    main(args)
