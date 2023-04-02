import unittest
import logging
import numpy as np
import os
import sys

from pyDeepLearn.tests.test_utils import np_array_comp
from pyDeepLearn.TanhLayer import TanhLayer

test_dir_path = os.path.dirname(os.path.abspath( __file__ ))
test_data_path = os.path.join(test_dir_path, 'test_data', 'test_data.csv')

logging.basicConfig(
            format='%(asctime)s,%(msecs)d %(levelname)-8s ' +
                   '[%(filename)s:%(lineno)d] %(message)s',
            datefmt='%Y-%m-%d:%H:%M:%S',
            level=logging.INFO)

class TestTanhLayer(unittest.TestCase):
    def __init__(self, *args, **kwargs):
      if not os.path.isfile(test_data_path):
        logging.error(f"invalid test data path at {test_data_path}")
        sys.exit(1)
      self.input_data = np.genfromtxt(test_data_path,delimiter=',')
      self.layer = TanhLayer(self.input_data)
      super(TestTanhLayer, self).__init__(*args, **kwargs)

    def test_forward(self):
      output = self.layer.forward(self.input_data[0])
      expected_out = np.array([1.0, 0.0, 1.0, 0.0, 0.76159416, 0.9993293 ])
      self.assertTrue(np_array_comp(output, expected_out))
      logging.info(f"test_forward ran!")

    def test_gradient(self):
      logging.info(f"start test_gradient2 ")
      self.layer.forward(self.input_data[0])
      expected_out = np.array([[[1.00000000e-06, 1.00000100e+00, 1.00000000e-06,
                               1.00000100e+00, 4.19975342e-01, 1.34195068e-03]]])
      expected_out = expected_out * np.identity(len(expected_out[0][0]))

      output = self.layer.gradient()
      self.assertTrue(np_array_comp(output, expected_out))
      logging.info(f"test_gradient ran!")

    def test_gradient2(self):
      logging.info(f"start test_gradient2 ")
      self.layer.forward(self.input_data[:2])
      expected_out = np.array([[[1.00000000e-06, 1.00000100e+00, 1.00000000e-06,
                               1.00000100e+00, 4.19975342e-01, 1.34195068e-03]],
                               [[1.00000000e-06, 4.19975342e-01, 1.00000000e-06, 
                               4.19975342e-01, 1.00000100e+00, 9.86703717e-03]]])
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

      layer = TanhLayer(input_data)
      output = layer.forward(input_data)
      logging.debug(f"The Tanh layer returned \n" +
                   f"{np.array2string(output, separator=', ')}"
                  )
      expected_out = np.array( 
                    [[0.76159416, 0.96402758, 0.99505475, 0.9993293],
                    [0.9999092, 0.99998771, 0.99999834, 0.99999977]]
                  )
      self.assertTrue(np_array_comp(output, expected_out))
      logging.info(f"Complete: test_hw1")

    def test_hw2(self):
      print()
      logging.info(f"Starting: test_hw2")
      # Seed the random number generator for consistent results
      np.random.seed(0)
      
      # Create the input data array per the HW assignment
      input_data =np.array([[1.0, 2.0, 3.0, 4.0]])

      # Initialize the TanhLayer
      sizeOut = 2 # as defined in HW 1
      input_layer = TanhLayer(len(input_data[0]))
      
      # Run the Forward method and gradient on the data from HW assignment 2
      fwd = input_layer.forward(input_data)
      print(f"fwd {fwd}")
      output = input_layer.gradient()
      print(f"grad {output}")
      logging.debug(f"The Tanh layer grad returned \n" +
                   f"{np.array2string(output, separator=', ')}"
                  )
      expected_out = np.array(
         [[[0.41997534, 0.07065182, 0.00986704, 0.00134195]]]
        )
      expected_out = expected_out * np.identity(len(expected_out[0][0]))

      self.assertTrue(np_array_comp(output, expected_out))
      logging.info(f"Complete: test_hw2")

    def test_3(self):
      print()
      logging.info(f"Starting: test_3")
      # Seed the random number generator for consistent results
      np.random.seed(0)
      
      # Create the input data array per the HW assignment
      input_data =np.array([[1.0, 2.0, 3.0, 4.0],
                       [2.0, 3.0, 4.0, 5.0]])

      # Initialize the TanhLayer
      input_layer = TanhLayer(len(input_data[0]))
      
      # Run the Forward method and gradient on the data from HW assignment 2
      fwd = input_layer.forward(input_data)
      print(f"fwd {fwd}")
      output = input_layer.gradient()
      print(f"grad {output}")
      logging.debug(f"The Tanh layer grad returned \n" +
                   f"{np.array2string(output, separator=', ')}"
                  )
      expected_out = np.array(
         [[[0.41997534, 0.07065182, 0.00986704, 0.00134195]]]
        )

      expected_out = expected_out * np.identity(len(expected_out[0][0]))

      self.assertTrue(np_array_comp(output[0], expected_out))
      expected_out2 = np.array(
         [[[7.06508249e-02, 9.86603717e-03, 1.34095068e-03, 1.81583231e-04]]]
        )
      expected_out2 = expected_out2 * np.identity(len(expected_out[0][0]))
      self.assertTrue(np_array_comp(output[1], expected_out2))
      logging.info(f"Complete: test_3")

if __name__ == '__main__':
    unittest.main()