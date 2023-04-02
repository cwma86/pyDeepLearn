from abc import ABC, abstractmethod
import logging
import numpy as np
import sys
class Layer(ABC):
  """
    A class to represent a Layer of this deep learning module. This is the 
    primary base type of all layers including 
    input layers
    fully connected layers
    activation layers

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
  def __init__(self):
    self.__prevIn = []
    self.__prevOut = []
  
  def setPrevIn(self, dataIn):
    self.__prevIn= dataIn
  
  def setPrevOut(self, out):
    self.__prevOut= out

  def getPrevIn(self):
    return self.__prevIn
  
  def getPrevOut(self):
    return self.__prevOut
  
  def backward(self, gradIn):
    sg = self.gradient()
    try:
      grad = np.zeros((gradIn.shape[0],sg.shape[2]))
      for n in range(gradIn.shape[0]): #compute for each observation in batch
        grad[n,:] = gradIn[n,:]@sg[n,:,:]

    except IndexError:
      logging.error(f"Invalid shape")
      logging.info(f"gradIn.shape {gradIn.shape} sg.shape {sg.shape}")
      sys.exit(1)
    except RuntimeWarning:
      logging.info(f"Warn!")
    return grad
  
  @abstractmethod
  def forward(self, dataIn):
    pass
  @abstractmethod
  def gradient(self, dataIn):
    pass
