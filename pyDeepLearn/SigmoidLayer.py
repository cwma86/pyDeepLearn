import logging
import numpy as np

from pyDeepLearn.LayerInterface import Layer
class SigmoidLayer(Layer):
  """
    A class to represent a Layer that provides a sigmoid modification to input
    data and scales its values between the ranges of 0 and 1
    ...

    Attributes
    ----------
    __prevIn : numpy matrix
        previously input data to the layer
    __prevOut : numpy matrix
        previously output data from the layer


    Methods
    -------
    forward():
        Method for back propagation of layer
    gradient():
        Method for calculating the gradient of the layer given its current
        prev input and output data
    backward():
        Method for back propagation of layer
  """
  def __init__(self, dataIn):
    super().__init__()


  def forward(self, dataIn):
    # Input data validation checks
    if dataIn.ndim == 1:
      dataIn = np.array([dataIn])
    if dataIn.ndim > 2:
      logging.error(f"invalid input data matrix dimensions {dataIn.ndim }")
      raise TypeError
    self.setPrevIn(dataIn)
    Y = 1/(1+np.exp(-dataIn))
    self.setPrevOut(Y)
    return Y

  def gradient(self):
    E = 0.00000001
    dj = np.zeros((self.getPrevOut().shape[0],
                   self.getPrevOut().shape[1],
                   self.getPrevOut().shape[1]))
    for k in range(self.getPrevOut().shape[0]):
      for i in range(self.getPrevOut().shape[1]):
        for j in range(self.getPrevOut().shape[1]):
          if i == j:
            dj[k][i][j] = self.getPrevOut()[k][j] * (1 - self.getPrevOut()[k][j])

    return dj