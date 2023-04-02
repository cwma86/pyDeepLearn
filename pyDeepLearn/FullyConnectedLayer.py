import logging
import math
import numpy as np
import sys

from pyDeepLearn.LayerInterface import Layer

class FullyConnectedLayer(Layer):
  """
    A class to represent a Fully connected layer provides a means of reducing
    or expanding data dimensionality

    ...

    Attributes
    ----------
    weights : numpy matrix
        Weight matrix used for modifying data dimensionality
    bias : numpy matrix
        bias matrix used for adding bias to the data
    __prevIn : numpy matrix
        previously input data to the layer
    __prevOut : numpy matrix
        previously output data from the layer


    Methods
    -------
    forward(additional=""):
        Method for back propagation of layer
    gradient(additional=""):
        Method for calculating the gradient of the layer given its current
        prev input and output data
    backward(additional=""):
        Method for back propagation of layer
  """
  def __init__(self, sizeIn, sizeOut, 
               weight=None, bias=None, 
               weight_up_func="updateWeights",
               eta=0.001):
    super().__init__()

    w_size = [sizeIn, sizeOut]
    self.setPrevIn(np.zeros(sizeIn))
    self.setPrevOut(np.zeros((1,sizeOut)))
    self.eta = eta
    if weight == None:
      self.weights = np.random.uniform(-0.0001, 0.0001, size=w_size)
    else:
      self.weights = np.random.uniform(weight[0], weight[1], size=w_size)
    if bias == None:
      self.bias = np.random.uniform(-0.0001, 0.0001, size=sizeOut)
    else:
      self.bias = np.random.uniform(bias[0], bias[1], size=sizeOut)

    if weight_up_func == "adam_weight_update":
      self.s = 0
      self.r = 0
      self.p1=0.9
      self.p2=0.999
      self.sig=10e-8
      self.weight_up_func = self.adam_weight_update
    else:
      self.weight_up_func = self.updateWeights
    self.epoch = 0 

  def getWeights(self):
    return self.weights

  def setWeights(self, weights):
    self.weights = weights
  
  def getBias(self):
    return self.bias
  
  def setBias(self, bias):
    self.bias = bias

  def forward(self, dataIn):
    if dataIn.ndim == 1:
      dataIn = np.array([dataIn])
    if dataIn.ndim > 2:
      logging.error(f"invalid input data matrix dimensions {dataIn.ndim }")
      raise TypeError
    self.setPrevIn(dataIn)
    # print(f" dataIn { dataIn.shape}")
    # print(f" self.weights { self.weights.shape}")
    # print(f" self.bias { self.bias.shape}")
    try:
      h = dataIn @ self.weights + self.bias
    except RuntimeWarning:
      logging.info(f"Warn!")
    # print(f" h  { h .shape}")
    self.setPrevOut(h)
    return h

  def gradient(self):
    dj = []
    for i in range(len(self.getPrevIn())):
      dj.append(self.getWeights().T)
    dj = np.array(dj)
    array_sum = np.sum(dj)
    if np.isnan(array_sum):
      logging.error("gradient is nan")
      print(len(self.getPrevIn()))
      print(self.getWeights())
      sys.exit(1)
    return dj
  
  def updateWeights(self, gradIn, epoch=1):
    dJdw= (self.getPrevIn().T @ gradIn)/gradIn.shape[0]
    self.weights = self.weights - self.eta * dJdw
    
    # add jitter after so many epochs
    if epoch % 10 == 0:
      self.weights = self.weights * 0.999
      
    dJdb = np.sum(gradIn, axis = 0)/gradIn.shape[0]
    self.bias = self.bias - self.eta * dJdb

  def reccurentWeightUpdate(self, djdw, djdb):
    array_sum = np.sum(djdw)
    if np.isnan(array_sum):
      logging.error("gradient is nan")
      sys.exit(1)
    # add jitter after so many epochs
    self.weights = self.weights - self.eta * djdw
    if self.epoch % 750 == 0:
      self.weights = self.weights * 0.999
      self.epoch = 1
    self.epoch += 1


    # TODO not really sure what to do with the bias here...
    self.bias = self.bias - self.eta * djdb

  def adam_weight_update(self, gradIn, epoch=1):
    self.s = self.p1 * self.s + ((1-self.p1) * np.sum( gradIn, axis=0)/gradIn.shape[0])
    self.r = self.p2 * self.r + ((1-self.p2) * np.sum((gradIn * gradIn), axis=0)/gradIn.shape[0])

    temp_s = self.s/(1-self.p1**epoch)
    temp_r = self.r/(1-self.p2**epoch)
    self.weights = self.weights - self.eta *(temp_s/(np.sqrt(temp_r)+self.sig))
    dJdb = np.sum(gradIn, axis = 0)/gradIn.shape[0]
    self.bias = self.bias - self.eta * dJdb

class RecurrentFcLayer(FullyConnectedLayer):
  def __init__(self, sizeIn, sizeOut, 
               weight=None, bias=None, 
               weight_up_func="updateWeights",
               eta=0.001):
    super().__init__( sizeIn=sizeIn, sizeOut=sizeOut, 
               weight=weight, bias=bias, 
               weight_up_func=weight_up_func,
               eta=eta)
