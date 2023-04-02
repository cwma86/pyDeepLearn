import logging
import numpy as np 
import os
import sys
import unittest

from pyDeepLearn.tests.test_utils import np_array_comp
from pyDeepLearn.InputLayer import InputLayer

test_dir_path = os.path.dirname(os.path.abspath( __file__ ))
test_data_path = os.path.join(test_dir_path, 'test_data', 'test_data.csv')

logging.basicConfig(
            format='%(asctime)s,%(msecs)d %(levelname)-8s ' +
                   '[%(filename)s:%(lineno)d] %(message)s',
            datefmt='%Y-%m-%d:%H:%M:%S',
            level=logging.INFO)

class TestInputLayer(unittest.TestCase):
    def __init__(self, *args, **kwargs):
      if not os.path.isfile(test_data_path):
        logging.error(f"invalid test data path at {test_data_path}")
        sys.exit(1)
      self.input_data = np.genfromtxt(test_data_path,delimiter=',')
      self.layer = InputLayer(self.input_data)
      super(TestInputLayer, self).__init__(*args, **kwargs)

    def test_constructor1(self):
      expected_mean = np.array([36.88, 0.54, 30.8311, 0.86, 0.22, 2.56])
      self.assertTrue(np_array_comp(self.layer.meanX, expected_mean))
      expected_std = np.array( [15.30991409, 0.50345743, 5.9219359,
                                1.1430357, 0.41845196, 1.12775739])
      self.assertTrue(np_array_comp(self.layer.stdX, expected_std))
      logging.info(f"test_constructor1 ran!")

    def test_constructor2(self):
      # Create a matrix of all 3's for testing std=0
      input_data =np.ones((5,5)) * 3
      layer = InputLayer(input_data)
      expected_mean = np.array([3, 3, 3, 3, 3])
      self.assertTrue(np_array_comp(layer.meanX, expected_mean))
      expected_std = np.array( [1, 1, 1, 1, 1])
      self.assertTrue(np_array_comp(layer.stdX, expected_std))
      logging.info(f"test_constructor2 Ran!")

    def test_forward(self):
      zscore = self.layer.forward(self.input_data)
      zscore_truth_path = os.path.join(test_dir_path, 'test_data',
                                  'inputlayer_zscore_truth.csv')
      if not os.path.isfile(zscore_truth_path):
        logging.error(f"invalid test data path at {zscore_truth_path}")
        sys.exit(1)
      zscore_truth = np.genfromtxt(zscore_truth_path,delimiter=',')
      self.assertTrue(np_array_comp(zscore, zscore_truth))
      logging.info(f"test_forward ran!")

    def test_hw1(self):
      print()
      logging.info(f"Starting: test_hw1")
      # Initalize the input input matrix per HW 1
      input_data = np.array([[1.0, 2.0, 3.0, 4.0],
                            [5.0, 6.0, 7.0, 8.0]])

      layer = InputLayer(input_data)
      output = layer.forward(input_data)
      logging.debug(f"The input layer returned \n" +
                   f"{np.array2string(output, separator=', ')}"
                  )
      expected_out = np.array([[-0.70710678, -0.70710678, -0.70710678, -0.70710678],
                               [ 0.70710678,  0.70710678,  0.70710678,  0.70710678]])
      self.assertTrue(np_array_comp(output, expected_out))
      logging.info(f"Complete: test_hw1")

    def test_gradient(self):
      # TODO not implemented
      self.layer.gradient()
      logging.info(f"test 2 ran!")


if __name__ == '__main__':
    unittest.main()