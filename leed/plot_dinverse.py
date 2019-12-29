import argparse
import os

import cv2
import numpy as np
import matplotlib.pyplot as plt

from leed.detector import detect
from leed.utils import get_images_and_voltages


def setup_argument_parser(parser):
    """
    Set argument
    """
    parser.add_argument('--input-images-dir', help='input images directory', required=True)
    parser.add_argument('--input-voltages-path', help='input image, beam voltage csv file', required=True)
    parser.add_argument('--rprime', help='calculated rprime by calc-rprime')
    parser.add_argument('--output-image-path', help='output plot image path')
    # parser.add_argument('--kind', help='base type', choices=['Au', 'Ag', 'Cu'], required=True)
    # parser.add_argument('--surface', help='base surface', choices=['110', '111'], required=True)


def adj_angle(thetas, d_invs):
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


def plot_dinverse(input_images_dir, input_voltages_path, rprime, output_image_path=None):
    d_invs = np.array([])
    thetas = np.array([])

    image_paths, voltages = get_images_and_voltages(input_images_dir, input_voltages_path)

    for i in range(len(image_paths)):
        vector = detect(os.path.join(input_images_dir, image_paths[i]))

        if vector is not None:
            x, theta = cv2.cartToPolar(vector[:, 0], vector[:, 1])
            theta = np.pi - theta
            d = np.sqrt(150.4 / voltages[i]) * np.sqrt(x ** 2 + rprime ** 2) / x
            d_inv = 1 / d

            d_invs = np.append(d_invs, d_inv)
            thetas = np.append(thetas, theta)

    d_invs = d_invs.flatten()
    thetas = thetas.flatten()

    plt.polar(thetas, d_invs, 'o')
    if output_image_path:
        plt.savefig(output_image_path)
    else:
        plt.show()


def main(args):
    plot_dinverse(args.input_images_dir, args.input_voltages_path, float(args.rprime), args.output_image_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    setup_argument_parser(parser)
    args = parser.parse_args()
    main(args)
