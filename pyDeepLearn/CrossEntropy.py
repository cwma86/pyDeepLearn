import logging
import math
import numpy as np
from pyDeepLearn.objectiveFuncInterface import objectiveFuncInterface

class CrossEntropy(objectiveFuncInterface):
  """
    A class to represent a cross entropy objective function
    ...

    Methods
    -------
    eval(y, yhat):
        evaluate the cross entropy of the provided data
    gradient(y, yhat):
        Method for calculating the gradient of the layer given its current
        prev input and output data
  """
  def eval(self, y, yhat):
    if y.ndim == 1:
      y = np.array([y])
    if y.ndim > 2:
      logging.error(f"invalid input data matrix dimensions {y.ndim }")
      raise TypeError
    E = 0.0000001
    j = np.zeros((y.shape))
    for i in range(y.shape[0]):
      j[i] = -1 * y[i] * np.log(yhat[i].T+E)
    j = np.mean(j)
    return j

  def gradient(self, y, yhat):
    E = 0.00000001
    dj = -1*(y/(yhat+E))
    return dj