from pyDeepLearn.LayerInterface import Layer
import logging
import numpy as np

class ReLuLayer(Layer):
  """
    A class to represent a Layer that provides a rectified linear modification
    to the input data by providing a ramping function that limits the data to 
    a min value of zero

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
    if dataIn.ndim == 1:
      dataIn = np.array([dataIn])
    if dataIn.ndim > 2:
      logging.error(f"invalid input data matrix dimensions {dataIn.ndim }")
      raise TypeError

    self.setPrevIn(dataIn)
    # Initialize a equal size matrix of all 0 values
    zero_matrix = np.zeros(dataIn.shape)
    # Replace any negative with 0
    Y = np.maximum(zero_matrix, dataIn)
    self.setPrevOut(Y)
    return Y

  def gradient(self):
    dj = np.zeros((self.getPrevOut().shape[0],
                self.getPrevOut().shape[1],
                self.getPrevOut().shape[1]))
    for k in range(self.getPrevOut().shape[0]):
      for i in range(self.getPrevOut().shape[1]):
        for j in range(self.getPrevOut().shape[1]):
          if i == j:
            if self.getPrevIn()[k][j] < 0 :
              dj[k][i][j] = 0
            else :
              dj[k][i][j] = 1

    return dj