import unittest
import logging
import numpy as np
import os
import sys

from pyDeepLearn.tests.test_utils import np_array_comp
from pyDeepLearn.SoftmaxLayer import SoftmaxLayer

test_dir_path = os.path.dirname(os.path.abspath( __file__ ))
test_data_path = os.path.join(test_dir_path, 'test_data', 'test_data.csv')

logging.basicConfig(
            format='%(asctime)s,%(msecs)d %(levelname)-8s ' +
                   '[%(filename)s:%(lineno)d] %(message)s',
            datefmt='%Y-%m-%d:%H:%M:%S',
            level=logging.INFO)
            
class TestSoftmaxLayer(unittest.TestCase):
    def __init__(self, *args, **kwargs):
      if not os.path.isfile(test_data_path):
        logging.error(f"invalid test data path at {test_data_path}")
        sys.exit(1)
      self.input_data = np.genfromtxt(test_data_path,delimiter=',')
      self.layer = SoftmaxLayer(self.input_data)
      super(TestSoftmaxLayer, self).__init__(*args, **kwargs)

    def test_forward(self):
      output = self.layer.forward(self.input_data[0])
      expected_out = np.array(
        [[1.36370327e-04, 7.64055183e-13, 9.99863630e-01, 
        7.64055183e-13, 2.07691732e-12, 4.17159995e-11]]
      )
      self.assertTrue(np_array_comp(output, expected_out))
      logging.info(f"test_forward ran!")

    def test_forward_invalid_dim(self):
      input_data = np.array([[[[1.0, 2.0, 3.0, 4.0],
                      [5.0, 6.0, 7.0, 8.0]],
                      [[1.0, 2.0, 3.0, 4.0],
                      [5.0, 6.0, 7.0, 8.0]]]])
      try:
        output = self.layer.forward(input_data)
        self.assertTrue(False)
      except  TypeError:
        print("exception caught")
        self.assertTrue(True)
      logging.info(f"test_forward ran!")

    def test_gradient_1(self):
      logging.info(f"Starting gradient test 1")
      input_data = np.array([[1.0, 2.0, 3.0, 4.0],
                            [5.0, 5.0, 5.0, 5.0]])


      layer = SoftmaxLayer(input_data)
      output = layer.forward(input_data)
      print(f"forward: {output}")
      expected_out = np.array(
        [[[0.0320586, 0.08714432, 0.23688282, 0.64391426],
          [0.25, 0.25, 0.25, 0.25]]]
      )
      self.assertTrue(np_array_comp(output, expected_out))
      expected_out = np.array([[[ 0.03103085, -0.00279373, -0.00759413, -0.02064299],
                                [-0.00279373,  0.07955019, -0.02064299, -0.05611347],
                                [-0.00759413, -0.02064299,  0.18076935, -0.15253222],
                                [-0.02064299, -0.05611347, -0.15253222,  0.22928869]],
                                [[ 0.1875, -0.0625, -0.0625, -0.0625],
                                [-0.0625, 0.1875, -0.0625, -0.0625],
                                [-0.0625, -0.0625, 0.1875, -0.0625],
                                [-0.0625, -0.0625, -0.0625, 0.1875]]])
      output = layer.gradient()
      print(f"out {output}")
      self.assertTrue(np_array_comp(output, expected_out))
      logging.info(f"test_gradient ran!")

    def test_gradient_2(self):
      logging.info(f"Starting gradient test 2")
      self.layer.forward(self.input_data[0])
      expected_out = np.array(
        [[ 1.36351730e-04, -1.04194455e-16, -1.36351730e-04, -1.04194455e-16, -2.83229894e-16, -5.68882450e-15],
        [-1.04194455e-16,  7.64055183e-13, -7.63950988e-13, -5.83780322e-25, -1.58687944e-24, -3.18733256e-23],
        [-1.36351730e-04, -7.63950988e-13,  1.36351776e-04, -7.63950988e-13, -2.07663409e-12, -4.17103107e-11],
        [-1.04194455e-16, -5.83780322e-25, -7.63950988e-13,  7.64055183e-13, -1.58687944e-24, -3.18733256e-23],
        [-2.83229894e-16, -1.58687944e-24, -2.07663409e-12, -1.58687944e-24,  2.07691732e-12, -8.66406818e-23],
        [-5.68882450e-15, -3.18733256e-23, -4.17103107e-11, -3.18733256e-23, -8.66406818e-23,  4.17159995e-11]]
      )
      output = self.layer.gradient()
      self.assertTrue(np_array_comp(output, expected_out))
      logging.info(f"test_gradient ran!")

    def test_hw1(self):
      print()
      logging.info(f"Starting: test_hw1")
      # Initalize the input input matrix per HW 1
      input_data = np.array([[1.0, 2.0, 3.0, 4.0],
                            [5.0, 6.0, 7.0, 8.0]])

      layer = SoftmaxLayer(input_data)
      output = layer.forward(input_data)
      logging.debug(f"The softmax layer returned \n" +
                   f"{np.array2string(output, separator=', ')}"
                  )
      print(output)
      expected_out = np.array([[0.0320586, 0.08714432, 0.23688282, 0.64391426],
                               [0.0320586, 0.08714432, 0.23688282, 0.64391426]])
      self.assertTrue(np_array_comp(output, expected_out))
      output = layer.gradient()

      expected_out = np.array([[[ 0.03103085, -0.00279373, -0.00759413, -0.02064299],
                                [-0.00279373,  0.07955019, -0.02064299, -0.05611347],
                                [-0.00759413, -0.02064299,  0.18076935, -0.15253222],
                                [-0.02064299, -0.05611347, -0.15253222,  0.22928869]],
                                [[ 0.03103085, -0.00279373, -0.00759413, -0.02064299],
                                [-0.00279373,  0.07955019, -0.02064299, -0.05611347],
                                [-0.00759413, -0.02064299,  0.18076935, -0.15253222],
                                [-0.02064299, -0.05611347, -0.15253222,  0.22928869]]])
      print(output)
      self.assertTrue(np_array_comp(output, expected_out))
      logging.info(f"Complete: test_hw1")

    def test_hw2(self):
      print()
      logging.info(f"Starting: test_hw2")
      # Seed the random number generator for consistent results
      np.random.seed(0)
      
      # Create the input data array per the HW assignment
      input_data =np.array([[1.0, 2.0, 3.0, 4.0]])

      # Initialize the SoftmaxLayer
      input_layer = SoftmaxLayer(input_data)
      
      # Run the Forward method and gradient on the data from HW assignment 2
      input_layer.forward(input_data)
      output = input_layer.gradient()
      logging.debug(f"The Softmax Layer grad returned \n" +
                   f"{np.array2string(output, separator=', ')}"
                  )
      expected_out = np.array(
          [[ 0.03103085, -0.00279373, -0.00759413, -0.02064299],
          [-0.00279373,  0.07955019, -0.02064299, -0.05611347],
          [-0.00759413, -0.02064299,  0.18076935, -0.15253222],
          [-0.02064299, -0.05611347, -0.15253222,  0.22928869]]
        )
      self.assertTrue(np_array_comp(output, expected_out))
      logging.info(f"Complete: test_hw2")

    def test_hw4_theory1(self):
      input_data = np.array([[2, 0, 0, 0],
                            [0, 2, 0, 0],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1]])
      logging.debug(f"HW4 Theory 1 target class matrix: \n{input_data}")

      layer = SoftmaxLayer(input_data)
      output = layer.forward(input_data)

      logging.debug(f"HW4 Theory 1: \n{output}")
      expected_out = np.array( [[0.71123459, 0.09625514, 0.09625514, 0.09625514],
                                [0.09625514, 0.71123459, 0.09625514, 0.09625514],
                                [0.1748777, 0.1748777, 0.47536689, 0.1748777 ],
                                [0.1748777,  0.1748777, 0.1748777,  0.47536689]])
      self.assertTrue(np_array_comp(output, expected_out))
      logging.info(f"Complete: test_hw4")


if __name__ == '__main__':
    unittest.main()