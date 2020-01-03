import unittest

from leed.calc_rprime import calc_rprime


class TestCalcRprime(unittest.TestCase):
    '''
    test case of calc_rprime.py
    '''
    def test_Au110(self):
        ''' Test of 6P/Au(110) '''
        input_images_dir = './data/6P_Au110/image/Au110'
        input_voltages_path = './data/6P_Au110/image/Au110/voltages.csv'
        base_type = {'kind': 'Au', 'surface': '110'}
        expected_r = 601.69
        actual_r = calc_rprime(input_images_dir, base_type, input_voltages_path, isplot=False, manual_r=expected_r)
        print(base_type)
        print('expected: {}, actual: {}, diff: {}'.format(expected_r, actual_r, abs(expected_r - actual_r)))
        self.assertAlmostEqual(expected_r, actual_r, delta=10)

    def test_Au111(self):
        ''' Test of BrHPB/Au(111) '''
        input_images_dir = './data/BrHPB_Au111/second/Au111'
        input_voltages_path = './data/BrHPB_Au111/second/Au111/voltages.csv'
        base_type = {'kind': 'Au', 'surface': '111'}
        expected_r = 623.08
        actual_r = calc_rprime(input_images_dir, base_type, input_voltages_path, isplot=False, manual_r=expected_r)
        print(base_type)
        print('expected: {}, actual: {}, diff: {}'.format(expected_r, actual_r, abs(expected_r - actual_r)))
        self.assertAlmostEqual(expected_r, actual_r, delta=10)

    def test_Ag110(self):
        ''' Test of Coronene/Ag(110) '''
        input_images_dir = './data/Coronene_Ag110/image/Ag110'
        input_voltages_path = './data/Coronene_Ag110/image/Ag110/voltages.csv'
        base_type = {'kind': 'Ag', 'surface': '110'}
        expected_r = 575.24
        actual_r = calc_rprime(input_images_dir, base_type, input_voltages_path, isplot=False, manual_r=expected_r)
        print(base_type)
        print('expected: {}, actual: {}, diff: {}'.format(expected_r, actual_r, abs(expected_r - actual_r)))
        self.assertAlmostEqual(expected_r, actual_r, delta=10)

    def test_Ag111(self):
        ''' Test of Coronene/Ag(111) '''
        input_images_dir = './data/Coronene_Ag111/image/Ag111'
        input_voltages_path = './data/Coronene_Ag111/image/Ag111/voltages.csv'
        base_type = {'kind': 'Ag', 'surface': '111'}
        expected_r = 617.11
        actual_r = calc_rprime(input_images_dir, base_type, input_voltages_path, isplot=False, manual_r=expected_r)
        print(base_type)
        print('expected: {}, actual: {}, diff: {}'.format(expected_r, actual_r, abs(expected_r - actual_r)))
        self.assertAlmostEqual(expected_r, actual_r, delta=10)

    def test_Cu110(self):
        ''' Test of BV/Cu(110) '''
        input_images_dir = './data/BV_Cu110/image/Cu110'
        input_voltages_path = './data/BV_Cu110/image/Cu110/voltages.csv'
        base_type = {'kind': 'Cu', 'surface': '110'}
        expected_r = 637.95
        actual_r = calc_rprime(input_images_dir, base_type, input_voltages_path, isplot=False, manual_r=expected_r)
        print(base_type)
        print('expected: {}, actual: {}, diff: {}'.format(expected_r, actual_r, abs(expected_r - actual_r)))
        self.assertAlmostEqual(expected_r, actual_r, delta=10)

    def test_Cu111(self):
        ''' Test of Coronene/Cu(111) '''
        input_images_dir = './data/Coronene_Cu111/image/Cu111'
        input_voltages_path = './data/Coronene_Cu111/image/Cu111/voltages.csv'
        base_type = {'kind': 'Cu', 'surface': '111'}
        expected_r = 603.2
        actual_r = calc_rprime(input_images_dir, base_type, input_voltages_path, isplot=False, manual_r=expected_r)
        print(base_type)
        print('expected: {}, actual: {}, diff: {}'.format(expected_r, actual_r, abs(expected_r - actual_r)))
        self.assertAlmostEqual(expected_r, actual_r, delta=10)


if __name__ == "__main__":
    unittest.main()
