import copy
import logging
import math
import matplotlib.pyplot as plt
import numpy as np
import sys

from pyDeepLearn.FullyConnectedLayer import FullyConnectedLayer
from pyDeepLearn.FullyConnectedLayer import RecurrentFcLayer


def test_train_split(input_data, test_percent=.33, shuffle=True):
  if shuffle:
    np.random.shuffle(input_data)
  test_size = round(len(input_data)*test_percent)
  test = input_data[:test_size, :]
  train = input_data[test_size:, :]
  return train, test 


def forward_layers(layers, x):
  h = x
  for i in range(len(layers)-1):
    h = layers[i].forward(h)
  return h

def forward_calc_error(layers, x, y, error_data=[], j=0):
  h = forward_layers(layers, x)
  # check performance of test layers
  error_data.append({'index': j,
                    'pred': h,
                    'obj_func': layers[-1].eval(y, h)})
  return error_data

def backward_layers(layers, x, y, epoch=0):
  grad = layers[-1].gradient(y, x)
  for i in range(len(layers)-2,0,-1):
    newgrad = layers[i].backward(grad)
    if(isinstance(layers[i], FullyConnectedLayer)):
      array_sum = np.sum(grad)
      if np.isnan(array_sum):
        logging.error(f"gradient is nan for layer: {i}")
        sys.exit(1)
      layers[i].weight_up_func(grad, epoch=epoch)
    grad = newgrad 
  
def run_layers(layers, 
               X_train, Y_train,
               X_test=None, Y_test=None,
               max_epoch=10000,
               error_exit=10e-10,
               num_batches=1):
  error_data = []
  error_data_test = []

  train_batch_size = X_train.shape[0]/num_batches
  # Run the for max max_epoch times
  counter = 1
  for j in range(1, max_epoch):
    for k in range(num_batches):
      train_start = math.floor(k*train_batch_size)
      if k == num_batches -1:
        train_end = X_train.shape[0]
      else:
        train_end = math.floor(k*train_batch_size+train_batch_size-1)
      x = X_train[train_start:train_end]
      y = Y_train[train_start:train_end]
      h = forward_layers(layers, x)
      backward_layers(layers, h, y, epoch=j)
      counter += 1
    # run the layers forward on the test data
    if not X_test is None:
      error_data_test = forward_calc_error(layers, X_test, Y_test, error_data_test, j)
    
    # run the layers forward on the train data
    error_data = forward_calc_error(layers, X_train, Y_train, error_data, j)

    if j > 1 and abs(error_data[-2]['obj_func'] - error_data[-1]['obj_func'] ) < error_exit:
      logging.info(f"error_data term met after {j} delta value: {error_data[-2]['obj_func'] - error_data[-1]['obj_func']}")
      break
    if j > 1 and j % 100 == 0:
      logging.debug(f"running j: {j} delta value: {error_data[-2]['obj_func'] - error_data[-1]['obj_func']}")    # run training layers backwards
  return error_data, error_data_test

def run_plot_epoch_J(layers, 
                      X_train, Y_train,
                      X_test=None, Y_test=None,
                      max_epoch=10000,
                      error_exit=10e-100,
                      plot_name='obj_plot.png',
                      num_batches=1):

  error_data, error_data_test = run_layers(layers, 
               X_train, Y_train,
               X_test, Y_test,
               max_epoch=max_epoch,
               error_exit=error_exit,
               num_batches=num_batches)

  # Create arrays of test/training data errors
  index = []
  obj_func = []
  pred = []
  for d in error_data:
    index.append(d['index'])
    obj_func.append(d['obj_func'])
    pred.append(d['pred'])
  index_test = []
  obj_func_test = []
  pred_test = []
  for d in error_data_test:
    index_test.append(d['index'])
    obj_func_test.append(d['obj_func'])
    pred_test.append(d['pred'])

  # plot test and train RMSE
  fig = plt.figure()
  ax = fig.add_subplot()
  ax.plot(index , obj_func, label="Log Loss training")
  ax.plot(index_test , obj_func_test, label="Log Loss test")
  ax.set_ylabel('obj func')
  ax.set_xlabel('epoch')
  ax.legend()
  plt.title("obj_plot")
  plt.savefig(plot_name)
  plt.clf()
  plt.cla()
  plt.close()

def one_hot_encode(Y):
  unqiue_classes = np.unique(Y)
  return multi_class_target_conv(Y, len(unqiue_classes) )

def multi_class_target_conv(Y, num_classes):
  mat = np.zeros((len(Y),num_classes))
  for i in range(len(Y)):
    mat[i][Y[i]] = 1
  return mat

def calc_acc_prec_rec(Y_true, Y_pred):
  correct_pred = np.count_nonzero(Y_true == Y_pred)
  accuracy = (correct_pred)/(len(Y_true))
  tp = 0 
  tn = 0 
  fp = 0 
  fn = 0 
  for i in range(len(Y_true)):
    if Y_true[i] == 1 and Y_pred[i] == 1:
      tp += 1
    elif Y_true[i] == 0 and Y_pred[i] == 0:
      tn += 1
    elif Y_true[i] == 0 and Y_pred[i] == 1:
      fp += 1
    elif Y_true[i] == 1 and Y_pred[i] == 0:
      fn += 1
  try:
    precision = tp/(tp+fp)
  except ZeroDivisionError:
    precision = 0.0 
  try:
    recall = tp/(tp+fn)
  except ZeroDivisionError:
    recall = 0.0
  print(f"accuracy {accuracy} precision {precision} recall {recall}")

"""
    forward propogate a set of data for a specific time
    
    to enable batch processing, the expected data format that 
    includes data from multiple data sets at a given time
    x = [data 1 feature 1 @ time 1, data 1 feature 2 @ time 1, ... data 1 feature n @ time 1],
                   [data 2 feature 1 @ time 1, data 2 feature 2 @ time 1, ... data 2 feature n @ time 1]
"""
def RNN_forward(layers, x):
  h = x
  for i in range(len(layers)-1):
    if(isinstance(layers[i], RecurrentFcLayer)):
      # Don't update h, for reccurent layer
      layers[i].forward(h)
      continue
    h = layers[i].forward(h)

    if (((i+2) < len(layers) -1 )and
          isinstance(layers[i+2], RecurrentFcLayer)):
      # this layers output should be summed with the 
      # previous reccurent layer
      h = h + layers[i+2].getPrevOut()
  return h

def reccurent_call(t, layer_hist, grad, djdw,  djdu, djdwb=0, djdub=0):
    if t < 0:
      return djdw, djdu, djdwb, djdub
    curr_layer = layer_hist[t]
    if t > 0:
      prev_layer = layer_hist[t-1]
      djdw += (prev_layer[-4].getPrevIn().T @ grad)/grad.shape[0]
      djdwb += np.sum(grad, axis = 0)/grad.shape[0]
    djdu += (curr_layer[-6].getPrevIn().T @ grad)/grad.shape[0]
    djdub += np.sum(grad, axis = 0)/grad.shape[0]
    grad = curr_layer[-4].backward(grad)
    t -= 1
    return reccurent_call(t, layer_hist, grad, djdw, djdu, djdwb, djdub)

"""
    back propogate the RNN for training

    expected inputs
    layer_hist = a list of layers that provide layer state at each iteration of the 
                 forward process
    y = truth data the objective function gradient

"""
def RNN_backward(layer_hist, 
                 y):
  djdv = 0
  djdw = 0
  djdu = 0
  djdvb = 0
  djdwb = 0
  djdub = 0
  grad = 0 
  for t in range(len(layer_hist)):
    layer = layer_hist[t]
    y_at_t = y[t]
    h = layer[-2].getPrevOut()
    djdy = layer[-1].gradient(y_at_t,h)
    dydh3 = layer[-2].backward(djdy)
    djdv += (layer[-3].getPrevIn().T @ dydh3)/dydh3.shape[0]
    djdvb += np.sum(dydh3, axis = 0)/dydh3.shape[0]
    grad = layer[-3].backward(dydh3)
    time = t
    djdw, djdu, djdwb, djdub = reccurent_call(time, layer_hist, grad, 
                                              djdw, djdu, 
                                              djdwb=djdwb, djdub=djdub)
    array_sum = np.sum(djdw)
    if np.isnan(array_sum):
      logging.error("gradient is nan")
    array_sum = np.sum(djdu)
    if np.isnan(array_sum):
      logging.error("gradient is nan")
    array_sum = np.sum(djdv)
    if np.isnan(array_sum):
      logging.error("gradient is nan")
  return np.array(djdv), np.array(djdw), np.array(djdu), np.array(djdvb), np.array(djdwb), np.array(djdub)


"""
    Validates the correct number of inputs (3)
    expected data format
    X_data = [times series data set 1, time series data set2, ... timeseries data set n]
    times series data = [data at time 1, data at time 2, ... data at time n]
    data at time = [data feature 1, data feature 2, ... data feature n]
"""
def validate_input_dims(X_data ):
    if X_data.ndim != 3:
      logging.warning(f"incorrect number of dims {X_data.ndim}")
    if X_data.ndim == 1:
      X_data = np.array([X_data])
    if X_data.ndim == 2:
      X_data = np.array([X_data])
    return X_data

def init_RNN_layer(layers, X_data):
  X_data = validate_input_dims(X_data)
  for layer in layers:
    if (isinstance(layer, RecurrentFcLayer) ):
      layer.setPrevOut(np.zeros((X_data[:, 0].shape[0],
                  layer.getPrevOut().shape[1])))
  return layers


"""
    provide data to the model for prediction

    expected data format
    X_data = [times series data set 1, time series data set2, ... timeseries data set n]
    times series data = [data at time 1, data at time 2, ... data at time n]
    data at time = [data feature 1, data feature 2, ... data feature n]
"""
def RNN_predict(layers, X_data):
    h =[]
    layers = init_RNN_layer(layers, X_data)

    # iterate through each time step
    print(f"X_data {X_data}")
    for k in range(X_data.shape[1]):
      X = X_data[:, k]
      h.append(RNN_forward(layers, X))
    h = np.array(h)

    # put the data back into tensor order
    temp_h = []
    for i in range(h.shape[1]):
      temp_h.append(h[:,i,:])
    h = np.array(temp_h)
    return h


"""
    provide data to the model for training
    
    expected data format
    X_data = [times series data set 1, time series data set2, ... timeseries data set n]
    times series data = [data at time 1, data at time 2, ... data at time n]
    data at time = [data feature 1, data feature 2, ... data feature n]
"""
def RNN_train(layers,
            X_train, Y_train,
            X_test=np.array(None), Y_test=np.array(None),
            max_epoch=1000,
            error_exit=10e-10,
            plotname=None):
  X_train = validate_input_dims(X_train)

  j_hist = []
  j_test_hist = []
    
  for i in range(max_epoch):
    fc_grads = []
    layer_hist = []
    Y = []
    epoc_j_hist = []
    # iterate through each time step
    layers = init_RNN_layer(layers, X_train)
    for k in range(X_train.shape[1]):
      # Each tensor is a seperate time ordered data set
      # If more than one tensor is present we need to run all data together
      # at each data point
      # for example all data at t=1 needs to be forwarded together
      X = X_train[:, k]
      Y.append(Y_train[:, k])
      RNN_forward(layers, X)
      layer_hist.append(copy.deepcopy(layers))
      epoc_j_hist.append(layers[-1].eval(Y_train[:, k], layers[-2].getPrevOut()))

    Y = np.array(Y)
    fc_grads.append(RNN_backward(layer_hist, Y))
    epoc_j_hist = np.array(epoc_j_hist)
    j_hist.append(np.mean(epoc_j_hist))
    # store the objective result for the last time stamp
    # Update the weights
    # start at the last index which is for the first FC layer
    fc_grad_index = len(fc_grads[0]) -4 
    for layer in layers:
      if(isinstance(layer, FullyConnectedLayer) or
        isinstance(layer, RecurrentFcLayer) ):
        layer.reccurentWeightUpdate(fc_grads[0][fc_grad_index], fc_grads[0][fc_grad_index+3])
        fc_grad_index -= 1

    if fc_grad_index != -1:
      logging.warning(f"something went wrong this index should be -1")
    if X_test.any() != None and Y_test.any() != None:
      h_test = RNN_predict(layers, X_test)
      epoc_j_hist_test = []
      for i in range(h_test.shape[0]):
        epoc_j_hist_test.append(layers[-1].eval(Y_test[i], h_test[i]))
      epoc_j_hist_test = np.array(epoc_j_hist_test)
      j_test_hist.append(np.mean(epoc_j_hist_test))

      for layer in layers:
        if(isinstance(layer, RecurrentFcLayer)):
          # Time series is complete, re-initialize reccurrent FC layer
          layer.setPrevOut(np.zeros(layer.getPrevOut().shape))

  plt.plot(range(len(j_hist)), j_hist, label="train J")
  plt.plot(range(len(j_test_hist)), j_test_hist, label="test J")
  plt.xlabel("epoch")
  plt.ylabel("least squares J")
  plt.legend()
  plt.savefig("project learn rate")
  if plotname: 
    plt.show()
  plt.clf()
  plt.cla()
  plt.close()
