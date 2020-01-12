import unittest
import warnings

from leed.calc_rprime import calc_rprime


class TestCalcRprime(unittest.TestCase):
    '''
    test case of calc_rprime.py
    '''
    def setUp(self):
        warnings.filterwarnings('ignore')
        self.error = 1.0  # allow less than 3.0% error

    def execute_calc(self, input_images_dir, base_type, input_voltages_path, expected_r, isplot=False):
        actual_r = calc_rprime(input_images_dir, base_type, input_voltages_path, isplot=isplot, manual_r=expected_r)
        diff_ratio = abs(actual_r / expected_r - 1) * 100

        print(base_type)
        print('expected: {}[px], actual: {}[px], diff ratio: {}[%]'.format(expected_r, actual_r, diff_ratio))
        return diff_ratio

    def test_Au110(self):
        ''' Test of 6P/Au(110) '''
        input_images_dir = './data/6P_Au110/image/Au110'
        input_voltages_path = './data/6P_Au110/image/Au110/voltages.csv'
        base_type = {'kind': 'Au', 'surface': '110'}

        expected_r = 601.69
        diff_ratio = self.execute_calc(input_images_dir, base_type, input_voltages_path, expected_r)
        self.assertLess(diff_ratio, self.error)

    def test_Au111(self):
        ''' Test of BrHPB/Au(111) '''
        input_images_dir = './data/BrHPB_Au111/second/Au111'
        input_voltages_path = './data/BrHPB_Au111/second/Au111/voltages.csv'
        base_type = {'kind': 'Au', 'surface': '111'}

        expected_r = 623.08
        diff_ratio = self.execute_calc(input_images_dir, base_type, input_voltages_path, expected_r)
        self.assertLess(diff_ratio, self.error)

    def test_Ag110(self):
        ''' Test of Coronene/Ag(110) '''
        input_images_dir = './data/Coronene_Ag110/image/Ag110'
        input_voltages_path = './data/Coronene_Ag110/image/Ag110/voltages.csv'
        base_type = {'kind': 'Ag', 'surface': '110'}

        expected_r = 575.24
        diff_ratio = self.execute_calc(input_images_dir, base_type, input_voltages_path, expected_r)
        self.assertLess(diff_ratio, self.error)

    def test_Ag111(self):
        ''' Test of Coronene/Ag(111) '''
        input_images_dir = './data/Coronene_Ag111/image/Ag111'
        input_voltages_path = './data/Coronene_Ag111/image/Ag111/voltages.csv'
        base_type = {'kind': 'Ag', 'surface': '111'}

        expected_r = 617.11
        diff_ratio = self.execute_calc(input_images_dir, base_type, input_voltages_path, expected_r)
        self.assertLess(diff_ratio, self.error)

    def test_Cu110(self):
        ''' Test of BV/Cu(110) '''
        input_images_dir = './data/BV_Cu110/image/Cu110'
        input_voltages_path = './data/BV_Cu110/image/Cu110/voltages.csv'
        base_type = {'kind': 'Cu', 'surface': '110'}

        expected_r = 637.95
        diff_ratio = self.execute_calc(input_images_dir, base_type, input_voltages_path, expected_r)
        self.assertLess(diff_ratio, self.error)

    def test_Cu111(self):
        ''' Test of Coronene/Cu(111) '''
        input_images_dir = './data/Coronene_Cu111/image/Cu111'
        input_voltages_path = './data/Coronene_Cu111/image/Cu111/voltages.csv'
        base_type = {'kind': 'Cu', 'surface': '111'}

        expected_r = 603.2
        diff_ratio = self.execute_calc(input_images_dir, base_type, input_voltages_path, expected_r)
        self.assertLess(diff_ratio, self.error)


if __name__ == "__main__":
    unittest.main()
