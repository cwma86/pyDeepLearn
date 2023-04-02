import logging
import math
import numpy as np
from pyDeepLearn.LayerInterface import Layer

class SoftmaxLayer(Layer):
  """
    A class to represent a Layer that provides a softmax modification to the
    input data which scales the data to a probability distribution [0,1]
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
    
    # Calc the exp value (e^x) for each value in the input matrix
    exp_mat = np.zeros(dataIn.shape)
    for i in range(dataIn.shape[0]): # each row
      for j in range(dataIn.shape[1]): # each value in the row
        exp_mat[i][j] = math.exp(dataIn[i][j])

    # Create a matrix of each rows summed value
    exp_sums = np.sum(exp_mat, axis=1)

    # Calculate the soft max values 
    output_mat = np.zeros(exp_mat.shape)
    for i in range(exp_mat.shape[0]):
      for j in range(exp_mat.shape[1]):
        output_mat[i][j] = exp_mat[i][j]/exp_sums[i]
    self.setPrevOut(output_mat)
    return output_mat


  def gradient(self):
    dk = np.zeros((self.getPrevOut().shape[0],
                   self.getPrevOut().shape[1],
                   self.getPrevOut().shape[1]))
    for k in range(self.getPrevOut().shape[0]):
      for i in range(self.getPrevOut().shape[1]):
        for j in range(self.getPrevOut().shape[1]):
          if i == j:
            #calc diag
            dk[k][i][j] = self.getPrevOut()[k][j]*(1-self.getPrevOut()[k][j])
          else:
            #calc off diag
            dk[k][i][j] = -1 * self.getPrevOut()[k][i] * self.getPrevOut()[k][j]
            dk[k][j][i] = dk[k][i][j]
    return dk