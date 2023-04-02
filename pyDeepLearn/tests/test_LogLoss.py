import logging
import numpy as np 
import os
import sys
import unittest

from pyDeepLearn.tests.test_utils import np_array_comp

from pyDeepLearn.LogLoss import LogLoss

test_dir_path = os.path.dirname(os.path.abspath( __file__ ))
test_data_path = os.path.join(test_dir_path, 'test_data', 'test_data.csv')

logging.basicConfig(
            format='%(asctime)s,%(msecs)d %(levelname)-8s ' +
                   '[%(filename)s:%(lineno)d] %(message)s',
            datefmt='%Y-%m-%d:%H:%M:%S',
            level=logging.INFO)

class TestLogLoss(unittest.TestCase):
    def test_eval1(self):
      logging.info(f"Running test_eval1")
      obj_func = LogLoss()
      y = np.array([[0.0]])
      y_hat = np.array([[0.2]])
      output = obj_func.eval(y, y_hat)
      expected = 0.2231435513142097
      logging.debug(f"The log loss layer eval() returned \n" +
                   f"{output}"
                  )
      self.assertTrue(np_array_comp(output, expected))
      logging.info(f"test_eval1 ran!")
      
    def test_eval2(self):
      logging.info(f"Running test_eval2")
      obj_func = LogLoss()
      y = np.array([[0.0, 0.0]])
      y_hat = np.array([[0.2, 0.2]])
      output = obj_func.eval(y, y_hat)
      expected = 0.2231435513142097
      logging.debug(f"The log loss layer eval() returned \n" +
                   f"{output}"
                  )
      self.assertTrue(np_array_comp(output, expected))
      output = obj_func.gradient(y, y_hat)
      print(f"grad: {output}")
      logging.info(f"test_eval2 ran!")

    def test_eval3(self):
      logging.info(f"Running test_eval3")
      obj_func = LogLoss()
      y = np.array([[0.0, 0.0], [0.0, 0.0]])
      y_hat = np.array([[0.2, 0.2], [0.1, 0.1]])
      output = obj_func.eval(y, y_hat)
      expected = 0.1642519154304695
      logging.debug(f"The log loss layer eval() returned \n" +
                   f"{output}"
                  )
      self.assertTrue(np_array_comp(output, expected))
      output = obj_func.gradient(y, y_hat)
      print(f"grad: {output}")
      logging.info(f"test_eval3 ran!")

    def test_gradient(self):
      logging.info(f"Running test_gradient")
      obj_func = LogLoss()
      y = 0.0
      y_hat = 0.2
      output = obj_func.gradient(y, y_hat)
      expected = 1.2499921875488276
      logging.debug(f"The log loss layer gradient() returned \n" +
                   f"{output}"
                  )
      self.assertTrue(np_array_comp(output, expected))
      logging.info(f"test_gradiant ran!")


if __name__ == '__main__':
    unittest.main()