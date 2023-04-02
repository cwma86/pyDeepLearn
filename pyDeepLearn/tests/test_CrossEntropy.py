import logging
import numpy as np 
import os
import sys
import unittest

from pyDeepLearn.tests.test_utils import np_array_comp

from pyDeepLearn.CrossEntropy import CrossEntropy

test_dir_path = os.path.dirname(os.path.abspath( __file__ ))
test_data_path = os.path.join(test_dir_path, 'test_data', 'test_data.csv')

logging.basicConfig(
            format='%(asctime)s,%(msecs)d %(levelname)-8s ' +
                   '[%(filename)s:%(lineno)d] %(message)s',
            datefmt='%Y-%m-%d:%H:%M:%S',
            level=logging.INFO)

class TestCrossEntropy(unittest.TestCase):
    def test_eval1(self):
      logging.info(f"Running test_eval1")
      obj_func = CrossEntropy()
      y = np.array([1])
      y_hat = np.array([0.2])
      output = obj_func.eval(y, y_hat)
      print(f"output: {output}")
      expected = 1.6093879136840585
      logging.debug(f"The cross entropy eval returned \n" +
                   f"{np.array2string(output, separator=', ')}"
                  )
      self.assertTrue(output, expected)
      logging.info(f"test_eval1 ran!")

    def test_eval2(self):
      logging.info(f"Running test_eval2")
      obj_func = CrossEntropy()
      y = np.array([1, 0, 0])
      y_hat = np.array([0.2, 0.2, 0.6])
      output = obj_func.eval(y, y_hat)
      expected = 0.5364791374780751
      logging.debug(f"The cross entropy eval returned \n" +
                   f"{np.array2string(output, separator=', ')}"
                  )
      self.assertTrue(np_array_comp(output, expected))
      logging.info(f"test_eval2 ran!")

    def test_eval3(self):
      logging.info(f"Running test_eval3")
      obj_func = CrossEntropy()
      y = np.array([[1, 0, 0], [1, 1, 1]])
      y_hat = np.array([[0.2, 0.2, 0.6], [0.2, 0.2, 0.6]])
      output = obj_func.eval(y, y_hat)
      expected = 0.8898562824003357
      logging.debug(f"The cross entropy eval returned \n" +
                   f"{np.array2string(output, separator=', ')}"
                  )
      self.assertTrue(np_array_comp(output, expected))
      logging.info(f"test_eval3 ran!")

    def test_gradient1(self):
      logging.info(f"Running test_gradient1")
      obj_func = CrossEntropy()
      y = np.array([1])
      y_hat = np.array([0.2])
      output = obj_func.gradient(y, y_hat)
      expected = np.array(
        [[-4.9999975]]
      )
      logging.info(f"The cross entropy gradient() returned \n" +
                   f"{np.array2string(output, separator=', ')}"
                  )
      self.assertTrue(np_array_comp(output, expected))
      logging.info(f"test_gradient1 ran!")

    def test_gradient2(self):
      logging.info(f"Running test_gradient2")
      obj_func = CrossEntropy()
      y = np.array([[1, 0, 0]])
      y_hat = np.array([[0.2, 0.2, 0.6]])
      output = obj_func.gradient(y, y_hat)
      expected = np.array(
        [[-4.9999975, 0, 0]]
      )
      logging.debug(f"The cross entropy gradient() returned \n" +
                   f"{np.array2string(output, separator=', ')}"
                  )
      self.assertTrue(np_array_comp(output, expected))
      logging.info(f"test_gradient2 ran!")

    def test_gradient3(self):
      logging.info(f"Running test_eval3")
      obj_func = CrossEntropy()
      y = np.array([[1, 0, 0], [1, 1, 1]])
      y_hat = np.array([[0.2, 0.2, 0.6], [0.2, 0.2, 0.6]])
      output = obj_func.gradient(y, y_hat)
      expected = np.array([[-4.9999975, 0.0, 0.0,],
                   [-4.9999975, -4.9999975, -1.66666639]])
      logging.debug(f"The cross entropy eval returned \n" +
                   f"{np.array2string(output, separator=', ')}"
                  )
      self.assertTrue(np_array_comp(output, expected))
      logging.info(f"test_eval3 ran!")


if __name__ == '__main__':
    unittest.main()