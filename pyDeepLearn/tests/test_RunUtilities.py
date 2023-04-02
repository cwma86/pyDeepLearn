import logging
import numpy as np 
import os
import sys
import unittest

from pyDeepLearn.tests.test_utils import np_array_comp

from pyDeepLearn.RunUtilities import RNN_backward
from pyDeepLearn.RunUtilities import RNN_forward
from pyDeepLearn.RunUtilities import RNN_train
from pyDeepLearn.RunUtilities import RNN_predict
from pyDeepLearn.RunUtilities import run_layers
from pyDeepLearn.RunUtilities import test_train_split
from pyDeepLearn.InputLayer import InputLayer
from pyDeepLearn.FullyConnectedLayer import FullyConnectedLayer
from pyDeepLearn.FullyConnectedLayer import RecurrentFcLayer
from pyDeepLearn.LinearLayer import LinearLayer
from pyDeepLearn.SigmoidLayer import SigmoidLayer
from pyDeepLearn.LogLoss import LogLoss
from pyDeepLearn.LeastSquares import LeastSquares

test_dir_path = os.path.dirname(os.path.abspath( __file__ ))
test_data_path = os.path.join(test_dir_path, 'test_data', 'test_data.csv')

logging.basicConfig(
            format='%(asctime)s,%(msecs)d %(levelname)-8s ' +
                   '[%(filename)s:%(lineno)d] %(message)s',
            datefmt='%Y-%m-%d:%H:%M:%S',
            level=logging.INFO)

class TestRunUtilities(unittest.TestCase):
    # def test_test_train_split(self):
    #   logging.info(f"test_test_train_split started!")
    #   if not os.path.isfile(test_data_path):
    #     logging.error(f"invalid test data path at {test_data_path}")
    #     sys.exit(1)
    #   self.input_data = np.genfromtxt(test_data_path,delimiter=',')
    #   np.random.seed(0)
    #   train, test  = test_train_split(self.input_data, test_percent=0.05)
    #   print(test)
    #   expected = np.array([[23.0, 1.0, 17.385, 1.0, 0.0, 2.0],
    #                        [62.0, 0.0, 26.29, 0.0, 1.0, 3.0]])
    #   self.assertTrue(np_array_comp(test, expected))
    #   logging.info(f"test_test_train_split ran!")

    # def test_run_layers(self):
    #   logging.info(f"test_run_layers started!")
    #   if not os.path.isfile(test_data_path):
    #     logging.error(f"invalid test data path at {test_data_path}")
    #     sys.exit(1)
    #   self.input_data = np.genfromtxt(test_data_path,delimiter=',')
    #   np.random.seed(0)
    #   train, test  = test_train_split(self.input_data, test_percent=0.33)
    #   x_train = train[:, :-1]
    #   y_train = np.array([train[:, -1]]).T
    #   x_test = test[:, :-1]
    #   y_test = np.array([test[:, -1]]).T
    #   # initialize the layers
    #   L1 = InputLayer(x_train)
    #   L2 = FullyConnectedLayer(x_train.shape[1], y_train.shape[1])
    #   L3 = SigmoidLayer(x_train)
    #   L4 = LogLoss()
    #   layers = [L1, L2, L3, L4]
    #   error_data, error_data_test = run_layers(layers, 
    #            x_train, y_train,
    #            x_test, y_test)
    #   expected = np.array([[0.99999974],
    #                         [0.99999948],
    #                         [0.99999961],
    #                         [0.99999907],
    #                         [0.99999967],
    #                         [0.99999888],
    #                         [0.9999998 ],
    #                         [0.99999922],
    #                         [0.99999956],
    #                         [0.99999955],
    #                         [0.99999935],
    #                         [0.9999997 ],
    #                         [0.99999877],
    #                         [0.99999982],
    #                         [0.99999916],
    #                         [0.99999931]])
    #   test = error_data_test[-1]['pred']
    #   self.assertTrue(np_array_comp(test, expected))
    #   logging.info(f"test_run_layers ran!")

    # def test_RNN_forward(self):
    #   X_train = np.array([
    #                       [1.0]
    #                      ])
    #   Y_train = np.array([[2.0],[ 3.0]
    #                       ])
    #   # L1 = InputLayer(X_train)
    #   L2 = FullyConnectedLayer(X_train.shape[1], 
    #                           2,
    #                           eta=0.008)
    #   L2.setWeights(np.array([[-0.1, 0.8]]))
    #   L2.setBias(np.array([[0.0, 0.0]]))
    #   L3 = LinearLayer(X_train)
    #   L4 = RecurrentFcLayer(2, 
    #                            2,
    #                           eta=0.008)
    #   L4.setWeights(np.array([[0.3, 0.7],
    #                           [-0.9, 0.9]]))
    #   L4.setBias(np.array([[0.0, 0.0]]))
    #   L5 = FullyConnectedLayer(2, 
    #                           1,
    #                           eta=0.008)
    #   L5.setWeights(np.array([[0.6],
    #                           [0.9]]))
    #   L5.setBias(np.array([[0.0]]))
    #   L6 = LinearLayer(X_train)
    #   L7 = LeastSquares()
    #   layers = [L2, L3, L4, L5, L6, L7]
    #   for i in range(X_train.shape[0]):
    #     h = RNN_forward(layers, X_train[i])
    #     logging.info(f"h {h}")
    #   self.assertEqual(h.shape, (1,1))
    #   self.assertAlmostEqual(h[0][0], 0.66)
    #   x2 = np.array([[0]])
    #   logging.info(f"time2")
    #   for i in range(x2.shape[0]):
    #     h = RNN_forward(layers, x2[i])
    #     logging.info(f"h {h}")
    #   self.assertEqual(h.shape, (1,1))
    #   self.assertAlmostEqual(h[0][0], 0.135)
    #   print(f"U weight {L2.getWeights()}")
    #   print(f"w weight {L4.getWeights()}")
    #   print(f"v weight {L5.getWeights()}")

    def test_RNN_train(self):
      X_train = np.array([[
                          [1.0],
                          [0.0]
                          ]])
      Y_train = np.array([[
                          [2.0],
                          [ 3.0]
                          ]])

      # L1 = InputLayer(X_train)
      U = FullyConnectedLayer(X_train.shape[2], 
                              Y_train.shape[1],
                              eta=0.008)
      U.setWeights(np.array([[-0.1, 0.8]]))
      U.setBias(np.array([[0.0, 0.0]]))
      L1 = LinearLayer()
      W = RecurrentFcLayer(Y_train.shape[1], 
                           Y_train.shape[1],
                              eta=0.008)
      W.setWeights(np.array([[0.3, 0.7],
                              [-0.9, 0.9]]))
      W.setBias(np.array([[0.0, 0.0]]))
      V = FullyConnectedLayer(Y_train.shape[1],
                              Y_train.shape[1], 
                              eta=0.008)
      V.setWeights(np.array([[0.6],
                              [0.9]]))
      V.setBias(np.array([[0.0]]))
      L4 = LinearLayer()
      L5 = LeastSquares()
      layers = [U, L1, W, V, L4, L5]
      RNN_train( 
              layers,
              X_train,
              Y_train,
              max_epoch=2000)

    def test_RNN_train2(self):
      X_train = np.array([[
                      [1.0],
                      [2.0]
                      ],
                      [
                      [1.0],
                      [2.0]
                      ]])
      Y_train = np.array([[
                          [2.0],
                          [3.0]
                          ],
                          [
                          [2.0],
                          [3.0]
                          ]])
      print(f"X_train.shape { X_train[0].shape}")

      L0 = InputLayer( X_train[0])
      U = FullyConnectedLayer( X_train[0].shape[1], 
                              X_train[0].shape[1],
                              eta=0.001)
      # U.setWeights(np.array([[-0.1, 0.8]]))
      # U.setBias(np.array([[0.0, 0.0]]))
      L1 = LinearLayer()
      W = RecurrentFcLayer(X_train[0].shape[1], 
                           X_train[0].shape[1],
                              eta=0.001)
      # W.setWeights(np.array([[0.3, 0.7],
      #                         [-0.9, 0.9]]))
      # W.setBias(np.array([[0.0, 0.0]]))
      V = FullyConnectedLayer(X_train[0].shape[1], 
                              Y_train[0].shape[1],
                              eta=0.001)
      # V.setWeights(np.array([[0.6],
      #                         [0.9]]))
      # V.setBias(np.array([[0.0]]))
      L4 = LinearLayer()
      L5 = LeastSquares()
      layers = [ U, L1, W, V, L4, L5]
      RNN_train(layers,
                      X_train,
                      Y_train,
                      max_epoch=2000
                      )
      print(f"U weight {U.getWeights()}")
      print(f"w weight {W.getWeights()}")
      print(f"v weight {V.getWeights()}")

      h = RNN_predict(layers, np.array([[1.0]]))
      print(f"prediction {h}")

      h = RNN_predict(layers, np.array([[4.0]]))
      print(f"prediction {h}")

if __name__ == '__main__':
    unittest.main()