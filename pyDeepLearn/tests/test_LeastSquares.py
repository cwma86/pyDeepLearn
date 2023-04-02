import logging
import numpy as np 
import os
import sys
import unittest

from pyDeepLearn.tests.test_utils import np_array_comp

from pyDeepLearn.LeastSquares import LeastSquares

test_dir_path = os.path.dirname(os.path.abspath( __file__ ))
test_data_path = os.path.join(test_dir_path, 'test_data', 'test_data.csv')

logging.basicConfig(
            format='%(asctime)s,%(msecs)d %(levelname)-8s ' +
                   '[%(filename)s:%(lineno)d] %(message)s',
            datefmt='%Y-%m-%d:%H:%M:%S',
            level=logging.INFO)

class TestLeastSquares(unittest.TestCase):
    def test_eval1(self):
      logging.info(f"Running test_eval1")
      obj_func = LeastSquares()
      y = np.array([0.0])
      y_hat = np.array([0.2])
      output = obj_func.eval(y, y_hat)
      expected =  0.04000000000000001
      logging.debug(f"The Least squares eval returned \n" +
                   f"{output}"
                  )
      self.assertTrue(np_array_comp(output, expected))
      logging.info(f"test_eval1 ran!")

    def test_eval2(self):
      logging.info(f"Running test_eval2")
      obj_func = LeastSquares()
      y = np.array([[1.0],[1.0]])
      y_hat = np.array([[0.2],[0.2]])
      output = obj_func.eval(y, y_hat)
      expected =   1.2800000000000002
      logging.debug(f"The Least squares eval returned \n" +
                   f"{output}"
                  )
      self.assertTrue(np_array_comp(output, expected))
      output = obj_func.gradient(y, y_hat)
      print(f"grad: {output}")
      logging.info(f"test_eval2 ran!")

    def test_eval3(self):
      logging.info(f"Running test_eval3")
      obj_func = LeastSquares()
      y = np.array([[1.0, 2.0],[1.0, 2.0]])
      y_hat = np.array([[0.2, 0.1],[0.2, 0.1]])
      output = obj_func.eval(y, y_hat)
      expected =   3.645
      logging.debug(f"The Least squares eval returned \n" +
                   f"{output}"
                  )
      self.assertTrue(np_array_comp(output, expected))
      output = obj_func.gradient(y, y_hat)
      print(f"grad: {output}")
      logging.info(f"test_eval3 ran!")

    def test_gradient(self):
      logging.info(f"Running test_gradient")
      obj_func = LeastSquares()
      y = np.array([0.0])
      y_hat = np.array([0.2])
      output = obj_func.gradient(y, y_hat)
      expected = 0.4
      logging.debug(f"The Least squares gradient() returned \n" +
                   f"{output}"
                  )
      self.assertTrue(np_array_comp(output, expected))
      logging.info(f"test_gradient ran!")


if __name__ == '__main__':
    unittest.main()