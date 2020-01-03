import argparse
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
    parser.add_argument('--isplot', help='draw a scatter plot of sinÎ¸ and X', action='store_true')
    parser.add_argument('--output-image-path', help='output plot image path')
    parser.add_argument('--manual-r', help='calculated r by myself')


def plot_scatter(xs, sinthetas, base_type, manual_r, rprime, intercept, output_image_path):
    plt.scatter(sinthetas, xs)
    plt.xlim([0, 0.6])
    plt.ylim([0, 500])
    plt.xlabel(r"sin${\theta}$")
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
    import sys

    try:
        x = sinthetas / np.sqrt(1 - sinthetas ** 2)
        rprime, intercept = np.polyfit(x, xs, 1)

        # remove outlier
        # outlier = np.abs(rprime*x+intercept - xs) > 30
        # x = np.insert(x[~outlier], 0, 0)
        # xs = np.insert(xs[~outlier], 0, 0)
        # sinthetas = np.insert(sinthetas[~outlier], 0, 0)
        # rprime, intercept = np.polyfit(x, xs, 1)

    except np.linalg.LinAlgError:
        sys.exit("r'の値が収束しませんでした。異なるデータセットを用いて再度試してください。")

    return rprime, intercept


def recalculate_sintheta(xs, sinthetas, base_type):
    xs = np.array(xs)
    sinthetas = np.array(sinthetas)

    min_ratio = np.min(xs[1:] / sinthetas[1:])
    idx_remove = []

    if base_type['kind'] == 'Au' and base_type['surface'] == '110':
        cands_n = [1, 2, 2 * np.sqrt(2), 3]
    else:
        cands_n = [1, np.sqrt(2), np.sqrt(3), 2]

    for i in range(1, len(xs)):
        ratio = xs[i] / sinthetas[i]

        isfind = False
        for cand_n in cands_n:
            if min_ratio <= ratio / cand_n < min_ratio + 160:
                sinthetas[i] *= cand_n
                isfind = True
                break

        if not isfind:
            idx_remove.append(i)

    xs = np.delete(xs, idx_remove)
    sinthetas = np.delete(sinthetas, idx_remove)

    return xs, sinthetas


def calc_x_and_sintheta(vector, base_type, voltage):
    # Polar coordinate transformation
    points = np.array(cv2.cartToPolar(vector[:, 0], vector[:, 1])).reshape(2, len(vector))
    # Sort by magnitude
    points = points[:, points[0, :].argsort()]
    n_points = points.shape[1]

    xs = []
    sinthetas = []
    d = {'Cu': 3.61496, 'Ag': 4.0862, 'Au': 4.07864}

    for i in range(n_points - 1):
        x, theta = points[:, i]

        for j in range(i + 1, n_points):
            other_x, other_theta = points[:, j]

            ratio_x = other_x / x
            diff_theta = abs(np.pi - abs(theta - other_theta))

            if (ratio_x < 1.2 and diff_theta < 0.15):

                # At this point, it is difficult to find n, so recalculation will be performed later.
                if base_type['surface'] == '111':
                    theoretical_d = d[base_type['kind']] * np.sqrt(6) / 4
                    sintheta = np.sqrt(150.4 / voltage) / theoretical_d
                else:
                    sintheta = np.sqrt(150.4 / voltage) / d[base_type['kind']]
                    if base_type['kind'] == 'Au':
                        sintheta /= 2

                xs.append(np.mean([x, other_x]))
                sinthetas.append(sintheta)

                break

    return xs, sinthetas


def calc_rprime(input_images_dir, base_type, input_voltages_path, isplot=False, output_image_path=None, manual_r=None):
    image_paths, voltages = get_images_and_voltages(input_images_dir, input_voltages_path)

    xs = [0]
    sinthetas = [0]

    for i in range(len(image_paths)):
        vector = detect(os.path.join(input_images_dir, image_paths[i]))

        if vector is not None:
            xs_i, sinthetas_i = calc_x_and_sintheta(vector, base_type, voltages[i])
            xs.extend(xs_i)
            sinthetas.extend(sinthetas_i)

    xs, sinthetas = recalculate_sintheta(xs, sinthetas, base_type)
    rprime, intercept = fit_xs_and_sinthetas(xs, sinthetas)

    if isplot:
        plot_scatter(xs, sinthetas, base_type, manual_r, rprime, intercept, output_image_path)

    return rprime


def main(args):
    base_type = {'kind': args.kind, 'surface': args.surface}
    rprime = calc_rprime(args.input_images_dir, base_type, args.input_voltages_path,
                         isplot=args.isplot, output_image_path=args.output_image_path, manual_r=args.manual_r)
    print("rprime: {}".format(rprime))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    setup_argument_parser(parser)
    args = parser.parse_args()
    main(args)
