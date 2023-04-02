import unittest
import logging
import numpy as np
import os
import sys

from pyDeepLearn.tests.test_utils import np_array_comp
from pyDeepLearn.LinearLayer import LinearLayer

test_dir_path = os.path.dirname(os.path.abspath( __file__ ))
test_data_path = os.path.join(test_dir_path, 'test_data', 'test_data.csv')

logging.basicConfig(
            format='%(asctime)s,%(msecs)d %(levelname)-8s ' +
                   '[%(filename)s:%(lineno)d] %(message)s',
            datefmt='%Y-%m-%d:%H:%M:%S',
            level=logging.INFO)

class TestLinearLayer(unittest.TestCase):
    def __init__(self, *args, **kwargs):
      if not os.path.isfile(test_data_path):
        logging.error(f"invalid test data path at {test_data_path}")
        sys.exit(1)
      self.input_data = np.genfromtxt(test_data_path,delimiter=',')
      self.layer = LinearLayer()
      super(TestLinearLayer, self).__init__(*args, **kwargs)

    def test_forward(self):
      logging.info(f"start test_forward ran!")
      output = self.layer.forward(self.input_data[0])
      expected_out = np.array([19.0, 0.0, 27.9, 0.0, 1.0, 4.0])
      self.assertTrue(np_array_comp(output, expected_out))
      logging.info(f"test_forward ran!")

    def test_gradient(self):
      logging.info(f"start test_gradient ran!")
      self.layer.forward(self.input_data[0])
      output = self.layer.gradient()
      expected_out = np.array([[1.0, 1.0, 1.0, 1.0, 1.0, 1.0]])
      expected_out = expected_out * np.identity(len(expected_out[0]))
      print(f"output {output}")
      print(f"expected_out {expected_out}")
      self.assertTrue(np_array_comp(output, expected_out))
      logging.info(f"test_gradient ran!")

    def test_hw1(self):
      print()
      logging.info(f"Starting: test_hw1")
      # Initalize the input input matrix per HW 1
      input_data = np.array([[1.0, 2.0, 3.0, 4.0],
                            [5.0, 6.0, 7.0, 8.0]])

      layer = LinearLayer()
      output = layer.forward(input_data)
      logging.debug(f"The linear layer returned \n" +
                   f"{np.array2string(output, separator=', ')}"
                  )
      expected_out = np.array([[1.0, 2.0, 3.0, 4.0],
                               [5.0, 6.0, 7.0, 8.0]])
      self.assertTrue(np_array_comp(output, expected_out))
      grad = layer.gradient()
      print(f"grad: {grad}")
      logging.info(f"Complete: test_hw1")

    def test_hw2(self):
      print()
      logging.info(f"Starting: test_hw2")
      # Seed the random number generator for consistent results
      np.random.seed(0)
      
      # Create the input data array per the HW assignment
      input_data =np.array([[1.0, 2.0, 3.0, 4.0]])

      # Initialize the LinearLayer
      sizeOut = 2 # as defined in HW 1
      input_layer = LinearLayer()
      
      # Run the Forward method and gradient on the data from HW assignment 2
      input_layer.forward(input_data)
      output = input_layer.gradient()
      logging.debug(f"The Linear Layer grad returned \n" +
                   f"{np.array2string(output, separator=', ')}"
                  )
      expected_out = np.array(
         [[1.0, 1.0, 1.0, 1.0]]
        )
      expected_out = expected_out * np.identity(len(expected_out[0]))
      self.assertTrue(np_array_comp(output, expected_out))
      logging.info(f"Complete: test_hw2")


if __name__ == '__main__':
    unittest.main()