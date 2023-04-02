import logging
import numpy as np
from pyDeepLearn.objectiveFuncInterface import objectiveFuncInterface

class LogLoss(objectiveFuncInterface):
  """
    A class to represent a log loss objective function
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
    E = 1e-8
    j = np.zeros(y.shape[0])
    for i in range(y.shape[0]):
      j[i] = np.sum(-((y[i]*np.log(yhat[i] + E) +  (1-y[i]) * np.log(1-yhat[i] + E))),0)/y[i].shape[0]
    j = np.mean(j)
    return j

  def gradient(self, y, yhat):
    E = 1e-8
    dj = -1 * (y - yhat) / (yhat*(1-yhat)+ E)
    return dj