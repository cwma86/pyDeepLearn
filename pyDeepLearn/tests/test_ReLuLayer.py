import logging
import numpy as np 
import os
import sys
import unittest

from pyDeepLearn.tests.test_utils import np_array_comp
from pyDeepLearn.ReLuLayer import ReLuLayer

test_dir_path = os.path.dirname(os.path.abspath( __file__ ))
test_data_path = os.path.join(test_dir_path, 'test_data', 'test_data.csv')

logging.basicConfig(
            format='%(asctime)s,%(msecs)d %(levelname)-8s ' +
                   '[%(filename)s:%(lineno)d] %(message)s',
            datefmt='%Y-%m-%d:%H:%M:%S',
            level=logging.INFO)
class TestReLuLayer(unittest.TestCase):
    def __init__(self, *args, **kwargs):
      if not os.path.isfile(test_data_path):
        logging.error(f"invalid test data path at {test_data_path}")
        sys.exit(1)
      self.input_data = np.genfromtxt(test_data_path,delimiter=',')
      self.layer = ReLuLayer(self.input_data)
      super(TestReLuLayer, self).__init__(*args, **kwargs)

    def test_forward(self):
      output = self.layer.forward(self.input_data[0])
      expected_out = np.array([19.0, 0.0, 27.9, 0.0, 1.0, 4.0])
      self.assertTrue(np_array_comp(output, expected_out))
      logging.info(f"test_forward ran!")

    def test_forward_neg(self):
      input_arr = np.array([-1, 0.0, 27.9, 0.0, -500.0, 4.0])
      output = self.layer.forward(input_arr)
      expected_out = np.array([0.0, 0.0, 27.9, 0.0, 0.0, 4.0])
      self.assertTrue(np_array_comp(output, expected_out))
      logging.info(f"test_forward ran!")

    def test_hw1(self):
      print()
      logging.info(f"Starting: test_hw1")
      # Initalize the input input matrix per HW 1
      input_data = np.array([[1.0, 2.0, 3.0, 4.0],
                            [5.0, 6.0, 7.0, 8.0]])

      layer = ReLuLayer(input_data)
      output = layer.forward(input_data)
      logging.debug(f"The ReLu layer returned \n" +
                   f"{np.array2string(output, separator=', ')}"
                  )
      expected_out = np.array([[1.0, 2.0, 3.0, 4.0],
                               [5.0, 6.0, 7.0, 8.0]])
      self.assertTrue(np_array_comp(output, expected_out))
      logging.info(f"Complete: test_hw1")

    def test_gradient(self):
      input_arr = np.array([-1, 0.0, 27.9, 0.0, -500.0, 4.0])
      self.layer.forward(input_arr)
      expected_out = np.array([[[0.0, 1.0, 1.0, 1.0, 0.0, 1.0]]])
      expected_out = expected_out * np.identity(len(expected_out[0][0]))

      output = self.layer.gradient()
      print(output)
      self.assertTrue(np_array_comp(output, expected_out))
      logging.info(f"test_gradient ran!")

    def test_gradient2(self):
      input_arr = np.array([1.0, 20.0, 27.9, 1.0, 500.0, 4.0])
      self.layer.forward(input_arr)
      expected_out = np.array([[[1.0, 1.0, 1.0, 1.0, 1.0, 1.0]]])
      expected_out = expected_out * np.identity(len(expected_out[0][0]))
      output = self.layer.gradient()
      self.assertTrue(np_array_comp(output, expected_out))
      logging.info(f"test_gradient ran!")

    def test_hw2(self):
      print()
      logging.info(f"Starting: test_hw2")
      # Seed the random number generator for consistent results
      np.random.seed(0)
      
      # Create the input data array per the HW assignment
      input_data =np.array([[1.0, 2.0, 3.0, 4.0]])

      # Initialize the ReLuLayer
      sizeOut = 2 # as defined in HW 1
      input_layer = ReLuLayer(len(input_data[0]))
      
      # Run the Forward method and gradient on the data from HW assignment 2
      input_layer.forward(input_data)
      output = input_layer.gradient()
      logging.debug(f"The ReLu Layer grad returned \n" +
                   f"{np.array2string(output, separator=', ')}"
                  )
      expected_out = np.array(
         [[[1.0, 1.0, 1.0, 1.0]]]
        )
      expected_out = expected_out * np.identity(len(expected_out[0][0]))
      self.assertTrue(np_array_comp(output, expected_out))
      logging.info(f"Complete: test_hw2")

if __name__ == '__main__':
    unittest.main()