#!/usr/bin/env python3
"""Command line entry point that demonstrates the pyDeepLearn framework.

The script provides three demonstrations, all driven by command line
arguments:

* :func:`linear_reg_test1` -- trains a small network to smooth noisy position
  measurements from a single track.
* :func:`linear_reg_test2` -- trains on one track and evaluates on a second,
  different track.
* :func:`run_project` -- trains a recurrent network over many tracks to reduce
  measurement noise for a target moving at constant velocity.

Run ``python main.py --help`` to see the available arguments.
"""
import argparse
import logging
import matplotlib.pyplot as plt
import numpy as np
import os


# Import the relivent modules for connecting layers
from pyDeepLearn.InputLayer import InputLayer
from pyDeepLearn.FullyConnectedLayer import FullyConnectedLayer
from pyDeepLearn.FullyConnectedLayer import RecurrentFcLayer
from pyDeepLearn.LinearLayer import LinearLayer
from pyDeepLearn.LeastSquares import LeastSquares
from pyDeepLearn.RunUtilities import run_plot_epoch_J
from pyDeepLearn.RunUtilities import forward_layers
from pyDeepLearn.RunUtilities import RNN_train
from pyDeepLearn.RunUtilities import RNN_predict

logging.basicConfig(
            format='%(asctime)s,%(msecs)d %(levelname)-8s ' +
                   '[%(filename)s:%(lineno)d] %(message)s',
            datefmt='%Y-%m-%d:%H:%M:%S',
            level=logging.INFO)


def arg_parsing():
  """Parse the command line arguments for the demonstration script.

  Returns
  -------
  argparse.Namespace
      Parsed arguments with the attributes ``filepath1``, ``filepath2``,
      ``datadir``, ``verbose``, ``showplot``, and ``weights``.
  """
  parser = argparse.ArgumentParser(description='Run script for CS 615 HW4\n')
  parser.add_argument('-f1', '--filepath1', type=str, default="",
                      help="file path to csv input data")
  parser.add_argument('-f2', '--filepath2', type=str, default="",
                      help="file path 2  to csv input data")
  parser.add_argument('-d', '--datadir', type=str, default="",
                      help="directory with track data")
  parser.add_argument('-v', '--verbose', action='store_true',
                      help="add verbose logging")
  parser.add_argument('-p', '--showplot', action='store_true',
                      help="displays plot")
  parser.add_argument('-w', '--weights', action='store_true',
                      help="use the initialized weights")
  args = parser.parse_args()

  if args.verbose:
    logging.getLogger().setLevel(logging.DEBUG)
  return args


def calc_RMSE(y, y_hat):
  """Compute the root mean squared error between two arrays.

  Parameters
  ----------
  y : numpy.ndarray
      Reference / ground truth values.
  y_hat : numpy.ndarray
      Predicted values with the same shape as ``y``.

  Returns
  -------
  float
      The root mean squared error.
  """
  return np.sqrt(((y - y_hat) ** 2).mean())


def calc_MAPE(y, y_hat):
  """Compute the mean absolute percentage error between two arrays.

  Parameters
  ----------
  y : numpy.ndarray
      Reference / ground truth values (must be non-zero to avoid dividing by
      zero).
  y_hat : numpy.ndarray
      Predicted values with the same shape as ``y``.

  Returns
  -------
  float
      The mean absolute percentage error expressed as a fraction.
  """
  return np.mean(np.abs((y - y_hat) / y))


def linear_reg_test1(args):
  """Train a small network to smooth noisy measurements from one track.

  The data is split into train and test partitions, a feed-forward network is
  trained with :func:`pyDeepLearn.RunUtilities.run_plot_epoch_J`, and the RMSE
  of the predictions is compared against the raw measurements to show whether
  the model reduces measurement noise for a target moving at constant velocity.

  Parameters
  ----------
  args : argparse.Namespace
      Parsed command line arguments. ``args.filepath1`` must point to a CSV
      file containing ``time, x, y, z`` measurements followed by the true
      ``x, y, z`` position.

  Returns
  -------
  None
      Results are logged and plots are produced as a side effect.
  """
  logging.info("Running main project linear_reg_test1")
  # Valid the input args
  if not os.path.isfile(args.filepath1):
    logging.warning(f"provide file path 1 is invalid {args.filepath1}")
    logging.info("Not running linear_reg_test1")
    return
  input_data = np.genfromtxt(args.filepath1, delimiter=',',  dtype=float, skip_header=1)
  np.random.shuffle(input_data)
  logging.debug(f"input_data shape: {input_data.shape}")
  test_len = round(input_data.shape[0] * 0.25)
  test_data = input_data[:test_len]
  train_data = input_data[test_len:]
  logging.debug(f"test_data shape: {test_data.shape}")
  logging.debug(f"test_data shape: {train_data.shape}")

  # split train and test
  logging.debug(f"train_data : {train_data[0]}")
  X_train = train_data[:, :4]  # get first 4 cols time, x, y,and  z
  Y_train = train_data[:, 4:7]  # get first 3 cols of truth for x, y, and z
  logging.debug(f"X_train : {X_train[0]}")
  logging.debug(f"X_train shape: {X_train.shape}")
  logging.debug(f"Y_train: {Y_train[0]}")
  logging.debug(f"Y_train shape: {Y_train.shape}")
  X_test = test_data[:, :4]  # get first 4 cols time, x, y,and  z
  Y_test = test_data[:, 4:7]  # get first 3 cols of truth for x, y, and z

  # initialize the layers
  L1 = InputLayer()
  L2 = FullyConnectedLayer(X_train.shape[1],
                           X_train.shape[1],
                           eta=0.008)
  L3 = LinearLayer()
  L4 = FullyConnectedLayer(X_train.shape[1],
                           Y_train.shape[1],
                           eta=0.008)
  L5 = LinearLayer()
  L6 = LeastSquares()
  layers = [L1, L2, L3, L4, L5, L6]
  run_plot_epoch_J(layers,
                      X_train, Y_train,
                      X_test, Y_test,
                      max_epoch=2000,
                      error_exit=10e-8,
                      num_batches=1)
  Y_pred = forward_layers(layers, X_test)

  plt.scatter(X_test[:, 0], Y_pred[:, 0], c='r', label='predicted pos')
  plt.scatter(X_test[:, 0], Y_test[:, 0], c='b', label='true pos')
  plt.scatter(X_test[:, 0], X_test[:, 1], c='g', label='meas pos')
  plt.legend()
  if args.showplot:
    plt.show()
  plt.clf()
  plt.cla()
  plt.close()
  # See if our predicted measurements are able to reduce the noise
  # and improve upon the accuracy of our input data.
  rmse = calc_RMSE(Y_pred[:, 0], Y_test[:, 0])
  logging.info(f"predicted rmse: {rmse}")
  rmse = calc_RMSE(X_test[:, 1], Y_test[:, 0])
  logging.info(f"measured rmse: {rmse}")


def linear_reg_test2(args):
  """Train on one track and evaluate on a second, different track.

  Demonstrates that the linear-regression approach depends on learning a
  track's velocity: the network is trained on ``args.filepath1`` and evaluated
  on ``args.filepath2``, which has a different direction and velocity.

  Parameters
  ----------
  args : argparse.Namespace
      Parsed command line arguments. ``args.filepath1`` supplies the training
      track and ``args.filepath2`` the test track.

  Returns
  -------
  None
      Results are logged and plots are produced as a side effect.
  """
  logging.info("Running main project linear_reg_test2")
  # Valid the input args
  if not os.path.isfile(args.filepath1):
    logging.warning(f"provide file path 1 is invalid {args.filepath1}")
    logging.info("Not running linear_reg_test2")
    return
  train_data = np.genfromtxt(args.filepath1, delimiter=',',  dtype=float, skip_header=1)

  if not os.path.isfile(args.filepath2):
    logging.warning(f"provide file path 2 is invalid {args.filepath2}")
    logging.info("Not running linear_reg_test2")
    return
  test_data = np.genfromtxt(args.filepath2, delimiter=',',  dtype=float, skip_header=1)

  logging.debug(f"test_data shape: {test_data.shape}")
  logging.debug(f"test_data shape: {train_data.shape}")

  # split train and test
  logging.debug(f"train_data : {train_data[0]}")
  X_train = train_data[:, :4]  # get first 4 cols time, x, y,and  z
  Y_train = train_data[:, 4:7]  # get first 3 cols of truth for x, y, and z
  logging.debug(f"X_train : {X_train[0]}")
  logging.debug(f"X_train shape: {X_train.shape}")
  logging.debug(f"Y_train: {Y_train[0]}")
  logging.debug(f"Y_train shape: {Y_train.shape}")
  X_test = test_data[:, :4]  # get first 4 cols time, x, y,and  z
  Y_test = test_data[:, 4:7]  # get first 3 cols of truth for x, y, and z

  # initialize the layers
  L1 = InputLayer()
  L2 = FullyConnectedLayer(X_train.shape[1],
                           X_train.shape[1],
                           eta=0.008)
  L3 = LinearLayer()
  L4 = FullyConnectedLayer(X_train.shape[1],
                           Y_train.shape[1],
                           eta=0.008)
  L5 = LinearLayer()
  L6 = LeastSquares()
  layers = [L1, L2, L3, L4, L5, L6]
  run_plot_epoch_J(layers,
                      X_train, Y_train,
                      X_test, Y_test,
                      max_epoch=2000,
                      error_exit=10e-8,
                      num_batches=1)
  Y_pred = forward_layers(layers, X_test)

  plt.scatter(X_test[:, 0], Y_pred[:, 0], c='r', label='predicted pos')
  plt.scatter(X_test[:, 0], Y_test[:, 0], c='b', label='true pos')
  plt.scatter(X_test[:, 0], X_test[:, 1], c='g', label='meas pos')
  plt.legend()
  if args.showplot:
    plt.show()
  plt.clf()
  plt.cla()
  plt.close()
  # Show that the linear regression modeling approach is dependant
  # on its ability to learn a tracks velocity by attempting to predict
  # a new track moving at a different trajectory that the one used
  # to train the model
  rmse = calc_RMSE(Y_pred[:, 0], Y_test[:, 0])
  logging.info(f"predicted rmse: {rmse}")
  rmse = calc_RMSE(X_test[:, 1], Y_test[:, 0])
  logging.info(f"measured rmse: {rmse}")


def run_project(args):
  """Train a recurrent network to reduce measurement noise across many tracks.

  Loads 500 tracks for training and 200 for testing from ``args.datadir`` (each
  truncated to 20 time steps), builds a recurrent network, and trains it with
  :func:`pyDeepLearn.RunUtilities.RNN_train`. Predicted positions are then
  compared against the true and measured positions using RMSE.

  Parameters
  ----------
  args : argparse.Namespace
      Parsed command line arguments. ``args.datadir`` must point to a directory
      of track CSV files; ``args.weights`` selects whether to reuse saved
      weights instead of training the network.

  Returns
  -------
  None
      Results are logged and plots are produced as a side effect.
  """
  # Valid the input args
  if not os.path.isdir(args.datadir):
    logging.warning(f"provide datadir path  is invalid {args.datadir}")
    logging.info("Not running test_RNN2")
    return
  track_list = os.listdir(args.datadir)
  train_data = []
  for i in range(500):
    # get train tracks
    track_file = track_list.pop()
    track_file = os.path.join(args.datadir, track_file)
    train_data.append(np.genfromtxt(track_file, delimiter=',',  dtype=float, skip_header=1))
  train_data = np.array(train_data)

  test_data = []
  for i in range(200):
    # get train tracks
    track_file = track_list.pop()
    track_file = os.path.join(args.datadir, track_file)
    test_data.append(np.genfromtxt(track_file, delimiter=',',  dtype=float, skip_header=1))
  test_data = np.array(test_data)

  X_test = test_data[:, :20, 0:4]  # get first 4 cols time, x, y,and  z
  Y_test = test_data[:, :20, 4:10]  # get first 3 cols of truth for x, y, and z

  logging.debug(f"test_data shape: {test_data.shape}")
  logging.debug(f"test_data shape: {train_data.shape}")

  # split train and test
  logging.debug(f"train_data : {train_data[0]}")
  X_train = train_data[:, :20, 0:4]  # get first 4 cols time, x, y,and  z
  Y_train = train_data[:, :20, 4:10]  # get first 3 cols of truth for x, y, and z
  # X_train = np.array([[[1],[2],[3]]])
  # Y_train = np.array([[[1],[2],[3]]])
  print(f"X_train.shape {X_train[0].shape}")

  internal_nodes = X_train[0].shape[1] * 2

  U = FullyConnectedLayer(X_train[0].shape[1],
                          internal_nodes,
                          weight=[0.015, 0.015],
                          bias=[0, 0],
                          eta=5e-7)
  if args.weights:
    U.setWeights(np.array([[1.14185731e-05, -5.55325197e-05,  2.80120501e-05,
                              -4.92934280e-05,  1.63363973e-05,  2.10323246e-05,
                              5.33487598e-05,  2.22410442e-05],
                            [-1.82494836e-01,  5.84665024e-01,  2.14376524e-01,
                              4.21472557e-01,  2.15067540e-01, -8.57863886e-02,
                              -1.51242373e-01, -2.29643114e-01],
                            [-9.98889190e-02, -3.99580833e-01,  3.17197727e-01,
                              -1.58212847e-01,  3.74617218e-01,  1.22161695e-01,
                              5.08052162e-01,  1.25579993e-01],
                            [5.81959088e-01,  1.11582216e-01, -2.40874751e-01,
                              3.16185137e-02, -2.99847804e-01,  2.59438493e-01,
                              -6.55042097e-02,  4.01025446e-01]]))
  U.epoch = 250  # add jitter to weights

  L1 = LinearLayer()
  W = RecurrentFcLayer(internal_nodes,
                        internal_nodes,
                          weight=[-0.001, 0.001],
                          bias=[0, 0],
                          eta=1e-7)
  if args.weights:
    W.setWeights(np.array([[0.00827842,  0.00068953, -0.00504, -0.00169637, -0.006952,
                              0.00255498, -0.00273279,  0.00492689],
                            [-0.00073527,  0.01121723, -0.00199848,  0.00731403, -0.00160589,
                              -0.00220093, -0.00842425, -0.00352849],
                            [-0.00576294, -0.0018309,  0.0032397,  0.00035464,  0.00439497,
                              -0.000675,  0.00226475, -0.0020746],
                            [-0.0012122,  0.00704134, -0.00099397,  0.00497179, -0.00082401,
                              -0.00175984, -0.00424639, -0.00271367],
                            [-0.00556767, -0.00315321,  0.00481663, -0.00014093,  0.00593016,
                              -0.00202366,  0.00368656, -0.00342825],
                            [0.00339821, -0.0020027, -0.00068416, -0.00209379, -0.00242505,
                              0.00073743,  0.00108158,  0.00205025],
                            [-0.0027881, -0.0074777,  0.00377585, -0.00541685,  0.00348766,
                              0.0003563,  0.00730356,  0.00086324],
                            [0.00478615, -0.0040354, -0.00303826, -0.00324511, -0.00221072,
                              0.00315227,  0.00083638,  0.00393594]]))
  W.epoch = 500  # add jitter to weights
  V = FullyConnectedLayer(internal_nodes,
                          Y_train[0].shape[1],
                          weight=[0.3, 0.3],
                          bias=[0, 0],
                          eta=5e-7)
  if args.weights:
    V.setWeights(np.array([[1.43910694e-01,  2.28414793e-01,  9.06341861e-01,
                              5.64262569e-04,  4.46433956e-04,  5.14752030e-04],
                            [9.11757935e-01, -7.16692133e-02,  4.35801275e-01,
                              5.59946248e-04,  4.41278658e-04,  6.04348977e-04],
                            [5.41203251e-01,  6.45751553e-01,  8.26980536e-02,
                              4.08336809e-04,  5.33735969e-04,  4.97276298e-04],
                            [7.48857207e-01,  1.69881460e-01,  3.55323748e-01,
                              5.82293609e-04,  5.40447119e-04,  6.35852541e-04],
                            [5.42239151e-01,  7.02853999e-01,  2.37434204e-02,
                              4.59323271e-04,  6.09944425e-04,  5.58113891e-04],
                            [2.40916134e-01,  4.50384268e-01,  5.83551685e-01,
                              5.24983584e-04,  5.17903183e-04,  5.28426562e-04],
                            [1.75372647e-01,  8.36448047e-01,  2.58461939e-01,
                              4.75514617e-04,  6.21773865e-04,  5.27537750e-04],
                            [9.70321795e-02,  4.53670936e-01,  7.25336634e-01,
                              5.79601276e-04,  5.50301531e-04,  5.57434246e-04]]))

  L4 = LinearLayer()
  L5 = LeastSquares()
  layers = [U, L1, W, V, L4, L5]

  # initial weights trained with 500 tracks for 5000 epochs (took 25 mins)
  if not args.weights:
    RNN_train(layers,
                    X_train,
                    Y_train,
                    X_test=X_test, Y_test=Y_test,
                    max_epoch=5000,
                    plotname="testplot",
                    )

  # TODO pickel weights for reload instead of print
  print(f"U weight \n{repr(U.getWeights())}")
  print(f"w weight \n{repr(W.getWeights())}")
  print(f"v weight \n{repr(V.getWeights())}")

  h = RNN_predict(layers, X_train)

  # Calculate the rmse (skip the first 3 measurements so that we can establish velocity)
  rmse = calc_RMSE(h[:, 3:, :], Y_train[:, 3:, :])
  print(f"training predicted rmse {rmse}")
  rmse = calc_RMSE(X_train[:, 3:, 1:], Y_train[:, 3:, :3])
  print(f"training measured rmse {rmse}")
  X_test
  h = RNN_predict(layers, X_test)

  # Calculate the rmse (skip the first 3 measurements so that we can establish velocity)
  rmse = calc_RMSE(h[:, 3:, :], Y_test[:, 3:, :])
  print(f"test predicted rmse {rmse}")
  rmse = calc_RMSE(X_test[:, 3:, 1:], Y_test[:, 3:, :3])
  print(f"test measured rmse {rmse}")

  plt.figure()
  ax = plt.axes(projection='3d')

  for i in range(0, 1):
    print(i)
    ax.scatter3D(h[i, 3:, 0], h[i, 3:, 1], h[i, 3:, 2], label="predicted track")
    ax.scatter3D(X_test[i, 3:, 1], X_test[i, 3:, 2], X_test[i, 3:, 3], label="meas track")
    ax.plot(Y_test[i, 3:, 0], Y_test[i, 3:, 1], Y_test[i, 3:, 2], label="true track")
  plt.legend()
  plt.show()


if __name__ == '__main__':
  args = arg_parsing()
  linear_reg_test1(args)
  linear_reg_test2(args)
  run_project(args)
