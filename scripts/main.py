from datetime import datetime
import os

from tqdm import tqdm

from detector import detect


if __name__ == "__main__":
    ismultiple = False

    if ismultiple:
        dir_name = datetime.now().strftime('%Y%m%d_%H%M')
        dir_path = os.path.join('output/image/', dir_name)
        os.makedirs(dir_path, exist_ok=True)

        for i in tqdm(range(52)):
            filename = 'L16{}.tif'.format(450 + i)
            detect(filename, dir_path=dir_path, ismultiple=ismultiple)
    else:
        pass
        # detect_mole_blob(617.11,
        #                  ['L16495.tif', 'L16496.tif', 'L16498.tif', 'L16499.tif', 'L16500.tif', 'L16501.tif'],
        #                  [15.7, 43.4, 23.9, 42.3, 45.3, 51.1],
        #                  isfilename=True)
