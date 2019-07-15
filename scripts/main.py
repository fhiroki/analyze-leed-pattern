from datetime import datetime
import os

from tqdm import tqdm

from detector import detect, detect_base_blob


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
        # Test of 6P/Au(110)
        DATA_DIR = 'data/6P_Au110/image/Au110'
        base_type = {'kind': 'Au', 'surface': '110'}
        # detect(os.path.join(DATA_DIR, 'L16543.tif'), isplot=True)
        r = detect_base_blob(DATA_DIR, base_type,
                             [134.4, 123.1, 115.7, 83.8, 79.9, 68.8, 63.4, 58.9, 57.0, 37.8, 29.5],
                             isfilename=True, isplot=True, manual_r=601.69)

        # Test of BrHPB/Au(111)
        DATA_DIR = 'data/BrHPB_Au111/second/Au111'
        base_type = {'kind': 'Au', 'surface': '111'}
        # detect(os.path.join(DATA_DIR, 'L16701.tif'), isplot=True)
        # TODO - 電圧を調査。
        # r = detect_base_blob(DATA_DIR, base_type
        #                      [],
        #                      isfilename=True, isplot=True, manual_r=623.08)

        # Test of Coronene/Ag(110)
        DATA_DIR = 'data/Coronene_Ag110/image/Ag110'
        base_type = {'kind': 'Ag', 'surface': '110'}
        # detect(os.path.join(DATA_DIR, 'L15495.tif'), isplot=True)
        r = detect_base_blob(DATA_DIR, base_type,
                             [33.1, 35.4, 38.1, 41.4, 50.8, 56.9, 60.1, 65.6],
                             isfilename=True, isplot=True, manual_r=575.24)

        # Test of Coronene/Ag(111)
        DATA_DIR = 'data/Coronene_Ag111/image/Ag111'
        base_type = {'kind': 'Ag', 'surface': '111'}
        # detect(os.path.join(DATA_DIR, 'L16471.tif'), isplot=True)
        r = detect_base_blob(DATA_DIR, base_type,
                             [80.6, 94.7, 109.2, 122.9, 136.0, 150.9, 159.1,
                                 179.2, 193.7, 215.3, 230.3, 252.7, 264.9],
                             image_paths=['L16469.tif', 'L16470.tif', 'L16471.tif', 'L16472.tif', 'L16473.tif',
                                          'L16474.tif', 'L16475.tif', 'L16476.tif', 'L16477.tif', 'L16478.tif',
                                          'L16479.tif', 'L16480.tif', 'L16481.tif'],
                             isfilename=True, isplot=True, manual_r=617.11)

        # Test of BV/Cu(110)
        DATA_DIR = 'data/BV_Cu110/image/Cu110'
        base_type = {'kind': 'Cu', 'surface': '110'}
        # TODO - manual_rの値を調査。
        # detect(os.path.join(DATA_DIR, 'L14846.tif'), isplot=True)
        r = detect_base_blob(DATA_DIR, base_type,
                             [163.5, 131.1, 61.4, 139.8, 136.1, 118.1, 60.8, 154.2, 136.3, 121.2, 63.4],
                             isfilename=True, isplot=True)

        # Test of Coronene/Cu(111)
        DATA_DIR = 'data/Coronene_Cu111/image/Cu111'
        base_type = {'kind': 'Cu', 'surface': '111'}
        # detect(os.path.join(DATA_DIR, 'L4898.tif'), isplot=True)
        r = detect_base_blob(DATA_DIR, base_type,
                             [98.9, 131.6, 141.1, 177.9, 231.6, 321.2],
                             image_paths=['L4898.tif', 'L4899.tif', 'L4900.tif', 'L4901.tif', 'L4902.tif', 'L4903.tif'],
                             isfilename=True, isplot=True, manual_r=603.2)

        # detect_mole_blob(617.11,
        #                  ['L16495.tif', 'L16496.tif', 'L16498.tif', 'L16499.tif', 'L16500.tif', 'L16501.tif'],
        #                  [15.7, 43.4, 23.9, 42.3, 45.3, 51.1],
        #                  isfilename=True)
