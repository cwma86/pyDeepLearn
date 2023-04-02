import logging
import numpy as np 
import os
import sys
import unittest

from pyDeepLearn.tests.test_utils import np_array_comp

from pyDeepLearn.FullyConnectedLayer import FullyConnectedLayer

test_dir_path = os.path.dirname(os.path.abspath( __file__ ))
test_data_path = os.path.join(test_dir_path, 'test_data', 'test_data.csv')

logging.basicConfig(
            format='%(asctime)s,%(msecs)d %(levelname)-8s ' +
                   '[%(filename)s:%(lineno)d] %(message)s',
            datefmt='%Y-%m-%d:%H:%M:%S',
            level=logging.INFO)

class TestFullyConnectedLayer(unittest.TestCase):
    def __init__(self, *args, **kwargs):
      if not os.path.isfile(test_data_path):
        logging.error(f"invalid test data path at {test_data_path}")
        sys.exit(1)
      self.input_data = np.genfromtxt(test_data_path,delimiter=',')

      # Seed the random number generator for consistent results
      np.random.seed(0)

      self.input_layer = FullyConnectedLayer(len(self.input_data [0]), 2)
      super(TestFullyConnectedLayer, self).__init__(*args, **kwargs)

    def test_constructor1(self):
      expected_weights = np.array([[ 9.76270079e-06,  4.30378733e-05],
                                  [ 2.05526752e-05,  8.97663660e-06],
                                  [-1.52690401e-05,  2.91788226e-05],
                                  [-1.24825577e-05,  7.83546002e-05],
                                  [ 9.27325521e-05, -2.33116962e-05],
                                  [ 5.83450076e-05,  5.77898395e-06]])
      self.assertTrue(np_array_comp(self.input_layer.weights, expected_weights))
      expected_bias = np.array( [1.36089122e-05, 8.51193277e-05])
      self.assertTrue(np_array_comp(self.input_layer.bias, expected_bias))
      logging.info(f"test_constructor1 ran!")

    def test_forward1(self):
      logging.info(f"test_forward1 runngin!")
      obs = self.input_layer.forward(self.input_data[0])
      obs_truth = np.array([9.92065900e-05, 1.71673231e-03])
      self.assertTrue(np_array_comp(obs, obs_truth))
      logging.info(f"test_forward1 ran!")

    def test_forward2(self):
      logging.info(f"test_forward2 runngin!")
      np.random.seed(0)
      input_data =np.array([[1.0]])
      sizeOut = 2 # as defined in HW 1
      input_layer = FullyConnectedLayer(len(input_data[0]), sizeOut)
      print(f"weight {input_layer.getWeights()}")
      print(f"bias {input_layer.getBias()}")
      obs = input_layer.forward(input_data)
      print(f"obs: {obs}")
      obs_truth = np.array( [[3.03153760e-05, 5.20145099e-05]])
      self.assertTrue(np_array_comp(obs, obs_truth))
      logging.info(f"test_forward2 ran!")

    def test_forward3(self):
      logging.info(f"test_forward3 runngin!")
      np.random.seed(0)
      input_data =np.array([[1.0, 2.0, 3.0]])
      sizeOut = 2 # as defined in HW 1
      input_layer = FullyConnectedLayer(len(input_data[0]), sizeOut)
      print(f"weight {input_layer.getWeights()}")
      print(f"bias {input_layer.getBias()}")
      obs = input_layer.forward(input_data)
      print(f"obs: {obs}")
      obs_truth = np.array([[-7.42162681e-06,  2.26882214e-04]])
      self.assertTrue(np_array_comp(obs, obs_truth))
      logging.info(f"test_forward3 ran!")

    def test_forward4(self):
      logging.info(f"test_forward4 runngin!")
      np.random.seed(0)
      input_data =np.array([[1.0, 2.0, 3.0],
                            [3.0, 4.0, 5.0]])
      sizeOut = 2 # as defined in HW 1
      input_layer = FullyConnectedLayer(len(input_data[0]), sizeOut)
      print(f"weight {input_layer.getWeights()}")
      print(f"bias {input_layer.getBias()}")
      obs = input_layer.forward(input_data)
      print(f"obs: {obs}")
      obs_truth = np.array([[-7.42162681e-06,  2.26882214e-04],
                            [ 2.26710448e-05, 3.89268879e-04]])
      self.assertTrue(np_array_comp(obs, obs_truth))
      logging.info(f"test_forward4 ran!")


    def test_hw1(self):
      print()
      logging.info(f"Starting: test_hw1")
      # Seed the random number generator for consistent results
      np.random.seed(0)
      
      # Create the input data array per the HW assignment
      input_data =np.array([[1.0, 2.0, 3.0, 4.0],
                           [5.0, 6.0, 7.0, 8.0]])

      # Initialize the FCL
      sizeOut = 2 # as defined in HW 1
      input_layer = FullyConnectedLayer(len(input_data[0]), sizeOut)
      
      # Run the Forward method on the data from HW assignment 1
      output = input_layer.forward(input_data)
      logging.debug(f"The fully connected layer returned \n" +
                   f"{np.array2string(output, separator=', ')}"
                  )
      expected_out = np.array([[4.78632519e-05, 4.38634319e-04],
                              [5.81183644e-05, 1.07682605e-03]]  )
      self.assertTrue(np_array_comp(output, expected_out))
      logging.info(f"Complete: test_hw1")

    def test_gradient1(self):
      self.input_layer.forward(self.input_data[0])
      obs_truth = np.array(
        [[ 9.76270079e-06,  2.05526752e-05, -1.52690401e-05, -1.24825577e-05, 9.27325521e-05, 5.83450076e-05],
        [ 4.30378733e-05,  8.97663660e-06, 2.91788226e-05, 7.83546002e-05, -2.33116962e-05, 5.77898395e-06]]
      )
      obs = self.input_layer.gradient()
      self.assertTrue(np_array_comp(obs, obs_truth))
      # TODO not implemented
      logging.info(f"test_gradient1 ran!")

    def test_gradient2(self):
      logging.info(f"test_gradient2 running!")
      np.random.seed(0)
      input_data =np.array([[1.0, 2.0, 3.0],
                            [3.0, 4.0, 5.0]])
      sizeOut = 2 # as defined in HW 1
      input_layer = FullyConnectedLayer(len(input_data[0]), sizeOut)
      print(f"weight {input_layer.getWeights()}")
      print(f"bias {input_layer.getBias()}")
      input_layer.forward(input_data)
      obs = input_layer.gradient()
      print(f"obs: {obs}")
      obs_truth = np.array( [[[ 9.76270079e-06, 2.05526752e-05, -1.52690401e-05],
                              [ 4.30378733e-05, 8.97663660e-06, 2.91788226e-05]],
                            [[ 9.76270079e-06, 2.05526752e-05, -1.52690401e-05],
                              [ 4.30378733e-05, 8.97663660e-06, 2.91788226e-05]]])
      self.assertTrue(np_array_comp(obs, obs_truth))

      self.assertTrue(np_array_comp(obs, obs_truth))
      # TODO not implemented
      logging.info(f"test_gradient2 ran!")

    def test_updateweights(self):
      logging.info(f"test_updateweights running!")
      np.random.seed(0)
      input_data =np.array([[1.0, 2.0, 3.0]])
      sizeOut = 2 # as defined in HW 1
      input_layer = FullyConnectedLayer(len(input_data[0]), sizeOut)
      print(f"weight {input_layer.getWeights()}")
      print(f"bias {input_layer.getBias()}")
      input_layer.forward(input_data)
      grad = input_layer.gradient()
      print(f"grad: {grad}")
      input_layer.updateWeights(np.array([[1.0,1.0]]))
      obs = input_layer.getWeights()
      print(f"obs: {obs}")
      obs_truth = np.array( [[-0.00099024, -0.00095696],
                            [-0.00197945, -0.00199102],
                            [-0.00301527, -0.00297082]])
      self.assertTrue(np_array_comp(obs, obs_truth))

      obs = input_layer.getBias()
      print(f"obs: {obs}")
      obs_truth = np.array([-0.00101248, -0.00092165])
      self.assertTrue(np_array_comp(obs, obs_truth))

      # TODO not implemented
      logging.info(f"test_updateweights ran!")


    def test_hw2(self):
      print()
      logging.info(f"Starting: test_hw2")
      # Seed the random number generator for consistent results
      np.random.seed(0)
      
      # Create the input data array per the HW assignment
      input_data =np.array([[1.0, 2.0, 3.0, 4.0]])

      # Initialize the FCL
      sizeOut = 2 # as defined in HW 1
      input_layer = FullyConnectedLayer(len(input_data[0]), sizeOut)
      
      # Run the Forward method and gradient on the data from HW assignment 2
      input_layer.forward(input_data)
      output = input_layer.gradient()
      logging.debug(f"The fully connected layer grad returned \n" +
                   f"{np.array2string(output, separator=', ')}"
                  )
      expected_out = np.array(
         [[ 9.76270079e-06, 2.05526752e-05, -1.52690401e-05, -1.24825577e-05],
          [ 4.30378733e-05, 8.97663660e-06, 2.91788226e-05, 7.83546002e-05]]
        )
      self.assertTrue(np_array_comp(output, expected_out))
      logging.info(f"Complete: test_hw2")

if __name__ == '__main__':
    unittest.main()