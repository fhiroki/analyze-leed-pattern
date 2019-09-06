from datetime import datetime
import os

from tqdm import tqdm

from detector import detect, detect_mole_blob


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
        # Test of Coronene/Cu(111)
        DATA_DIR = '../data/Coronene_Cu111/image/Coronene'
        detect_mole_blob(603.2, DATA_DIR,
                         [21.3, 8.0, 121.6, 66.1, 26.0, 143.4, 36.0, 25.1, 9.5, 141.6, 43.1],
                         isfilename=True)
        # detect_mole_blob(617.11, DATA_DIR,
        #                  ['L16495.tif', 'L16496.tif', 'L16498.tif', 'L16499.tif', 'L16500.tif', 'L16501.tif'],
        #                  [15.7, 43.4, 23.9, 42.3, 45.3, 51.1],
        #                  isfilename=True)
