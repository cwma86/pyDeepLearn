import logging
import numpy as np
from pyDeepLearn.objectiveFuncInterface import objectiveFuncInterface
class LeastSquares(objectiveFuncInterface):
  """
    A class to represent a least squares (square error) objective function
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
    j = np.mean((y-yhat).T @ (y - yhat))
    return j

  def gradient(self, y, yhat):
    dj = -2*(y-yhat)
    return dj