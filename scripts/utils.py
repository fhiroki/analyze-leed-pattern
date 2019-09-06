import os

import numpy as np


def sort_images_and_voltages(DATA_DIR, image_paths, voltages):
    if image_paths is None:
        image_paths = sorted(os.listdir(DATA_DIR))
        image_paths = [f for f in image_paths if f.endswith('tif')]

    # sort images and voltages by valtages
    image_paths = [x for _, x in sorted(zip(voltages, image_paths))]
    voltages = np.sort(voltages)
    return image_paths, voltages
