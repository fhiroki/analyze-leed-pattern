import unittest
from scripts.detector import detect_base_blob


DELTA = 10


class TestDetectBaseBlob(unittest.TestCase):
    '''
    test case of detector.py
    '''

    def test_Au110(self):
        ''' Test of 6P/Au(110) '''
        DATA_DIR = 'data/6P_Au110/image/Au110'
        base_type = {'kind': 'Au', 'surface': '110'}
        expected_r = 601.69
        actual_r = detect_base_blob(DATA_DIR, base_type,
                                    [134.4, 123.1, 115.7, 83.8, 79.9, 68.8, 63.4, 58.9, 57.0, 37.8, 29.5],
                                    isfilename=True, isplot=False, manual_r=expected_r)
        self.assertAlmostEqual(expected_r, actual_r, delta=DELTA)

    def test_Au111(self):
        ''' Test of BrHPB/Au(111) '''
        DATA_DIR = 'data/BrHPB_Au111/second/Au111'
        base_type = {'kind': 'Au', 'surface': '111'}
        expected_r = 623.08
        actual_r = detect_base_blob(DATA_DIR, base_type,
                                    [79.9, 95.9, 121.4, 163.8, 189.8, 256.1, 297.6, 346.9],
                                    isfilename=True, isplot=False, manual_r=expected_r)
        self.assertAlmostEqual(expected_r, actual_r, delta=DELTA)

    def test_Ag110(self):
        ''' Test of Coronene/Ag(110) '''
        DATA_DIR = 'data/Coronene_Ag110/image/Ag110'
        base_type = {'kind': 'Ag', 'surface': '110'}
        expected_r = 575.24
        actual_r = detect_base_blob(DATA_DIR, base_type,
                                    [33.1, 35.4, 38.1, 41.4, 50.8, 56.9, 60.1, 65.6],
                                    isfilename=True, isplot=False, manual_r=expected_r)
        self.assertAlmostEqual(expected_r, actual_r, delta=DELTA)

    def test_Ag111(self):
        ''' Test of Coronene/Ag(111) '''
        DATA_DIR = 'data/Coronene_Ag111/image/Ag111'
        base_type = {'kind': 'Ag', 'surface': '111'}
        expected_r = 617.11
        actual_r = detect_base_blob(DATA_DIR, base_type,
                                    [80.6, 94.7, 109.2, 122.9, 136.0, 150.9, 159.1,
                                     179.2, 193.7, 215.3, 230.3, 252.7, 264.9],
                                    isfilename=True, isplot=False, manual_r=expected_r)
        self.assertAlmostEqual(expected_r, actual_r, delta=DELTA)

    def test_Cu110(self):
        ''' Test of BV/Cu(110) '''
        DATA_DIR = 'data/BV_Cu110/image/Cu110'
        base_type = {'kind': 'Cu', 'surface': '110'}
        expected_r = 637.95
        actual_r = detect_base_blob(DATA_DIR, base_type,
                                    [163.5, 131.1, 61.4, 139.8, 136.1, 118.1, 60.8, 154.2, 136.3, 121.2, 63.4],
                                    isfilename=True, isplot=False, manual_r=expected_r)
        self.assertAlmostEqual(expected_r, actual_r, delta=DELTA)

    def test_Cu111(self):
        ''' Test of Coronene/Cu(111) '''
        DATA_DIR = 'data/Coronene_Cu111/image/Cu111'
        base_type = {'kind': 'Cu', 'surface': '111'}
        expected_r = 603.2
        actual_r = detect_base_blob(DATA_DIR, base_type,
                                    [98.9, 131.6, 141.1, 177.9, 231.6, 321.2],
                                    isfilename=True, isplot=False, manual_r=expected_r)
        self.assertAlmostEqual(expected_r, actual_r, delta=DELTA)


if __name__ == "__main__":
    unittest.main()
