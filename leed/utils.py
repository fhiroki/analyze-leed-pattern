import os

import pandas as pd


def get_images_and_voltages(input_images_dir, input_voltages_path):
    image_paths = sorted(os.listdir(input_images_dir))
    image_paths = [f for f in image_paths if f.endswith('.tif')]
    new_image_paths = []

    voltages_df = pd.read_csv(input_voltages_path)
    voltages = []
    for image_path in image_paths:
        image_number = int(image_path.split('.')[0][1:])
        voltage = voltages_df[voltages_df['image'] == image_number]['voltage'].values
        if voltage:
            new_image_paths.append(image_path)
            voltages.append(voltage[0])

    return new_image_paths, voltages
