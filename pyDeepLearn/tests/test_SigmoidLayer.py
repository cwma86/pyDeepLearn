import unittest
import logging
import numpy as np
import os
import sys

from pyDeepLearn.tests.test_utils import np_array_comp
from pyDeepLearn.SigmoidLayer import SigmoidLayer

test_dir_path = os.path.dirname(os.path.abspath( __file__ ))
test_data_path = os.path.join(test_dir_path, 'test_data', 'test_data.csv')

logging.basicConfig(
            format='%(asctime)s,%(msecs)d %(levelname)-8s ' +
                   '[%(filename)s:%(lineno)d] %(message)s',
            datefmt='%Y-%m-%d:%H:%M:%S',
            level=logging.INFO)

class TestSigmoidLayer(unittest.TestCase):
    def __init__(self, *args, **kwargs):
      if not os.path.isfile(test_data_path):
        logging.error(f"invalid test data path at {test_data_path}")
        sys.exit(1)
      self.input_data = np.genfromtxt(test_data_path,delimiter=',')
      self.layer = SigmoidLayer(self.input_data)
      super(TestSigmoidLayer, self).__init__(*args, **kwargs)

    def test_forward(self):
      output = self.layer.forward(self.input_data[0])
      expected_out = np.array([0.99999999, 0.5, 1.0, 0.5, 0.73105858, 0.98201379])
      self.assertTrue(np_array_comp(output, expected_out))
      logging.info(f"test_forward ran!")

    def test_gradient(self):
      logging.info(f"test_gradient start!")
      self.layer.forward(np.array([self.input_data[0]]))
      expected_out = np.array([[[5.60279642e-09, 2.50000000e-01, 7.64055486e-13, 
                               2.50000000e-01, 1.96611933e-01, 1.76627062e-02]]])
      expected_out = expected_out * np.identity(len(expected_out[0][0]))
      output = self.layer.gradient()
      self.assertTrue(np_array_comp(output, expected_out))
      logging.info(f"test_gradient ran!")

    def test_gradient2(self):
      logging.info(f"test_gradient start2!")
      self.layer.forward(self.input_data[0:2])
      print(self.input_data[0:2])
      expected_out = np.array([[[5.60279642e-09, 2.50000000e-01, 7.64055486e-13, 
                               2.50000000e-01, 1.96611933e-01, 1.76627062e-02]],
                               [[1.52299793e-08, 1.96611933e-01, 2.22044605e-15, 
                               1.96611933e-01, 2.50000000e-01, 4.51766597e-02]]])
      expected_out = expected_out * np.identity(len(expected_out[0][0]))

      output = self.layer.gradient()
      self.assertTrue(np_array_comp(output, expected_out))
      logging.info(f"test_gradient ran!")
      
    def test_hw1(self):
      print()
      logging.info(f"Starting: test_hw1")
      # Initalize the input input matrix per HW 1
      input_data = np.array([[1.0, 2.0, 3.0, 4.0],
                            [5.0, 6.0, 7.0, 8.0]])

      layer = SigmoidLayer(input_data)
      output = layer.forward(input_data)
      logging.debug(f"The sigmoid layer returned \n" +
                   f"{np.array2string(output, separator=', ')}"
                  )
      expected_out = np.array( [[0.73105858, 0.88079708, 0.95257413, 0.98201379],
                                [0.99330715, 0.99752738, 0.99908895, 0.99966465]])
      self.assertTrue(np_array_comp(output, expected_out))
      output = layer.gradient()
      print(f"grad {output}")
      logging.info(f"Complete: test_hw1")

    def test_hw2(self):
      print()
      logging.info(f"Starting: test_hw2")
      # Seed the random number generator for consistent results
      np.random.seed(0)
      
      # Create the input data array per the HW assignment
      input_data =np.array([[1.0, 2.0, 3.0, 4.0]])

      # Initialize the SigmoidLayer
      sizeOut = 2 # as defined in HW 1
      input_layer = SigmoidLayer(len(input_data[0]))
      
      # Run the Forward method and gradient on the data from HW assignment 2
      input_layer.forward(input_data)
      output = input_layer.gradient()
      logging.debug(f"The Sigmoid Layer grad returned \n" +
                   f"{np.array2string(output, separator=', ')}"
                  )
      expected_out = np.array(
          [[0.19661193, 0.10499359, 0.04517666, 0.01766271]]
        )
      expected_out = expected_out * np.identity(len(expected_out[0]))

      self.assertTrue(np_array_comp(output, expected_out))
      logging.info(f"Complete: test_hw2")


if __name__ == '__main__':
    unittest.main()