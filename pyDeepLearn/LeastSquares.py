"""Least squares (squared error) objective function."""
import logging
import numpy as np
from pyDeepLearn.objectiveFuncInterface import objectiveFuncInterface
class LeastSquares(objectiveFuncInterface):
  """Objective function computing the mean squared error of predictions.

  The scalar loss for a batch is ``J = mean((y - yhat)^T (y - yhat))`` and the
  gradient returned to the network is ``dJ/dyhat = -2 * (y - yhat)``.
  """
  def eval(self, y, yhat):
    """Evaluate the mean squared error for a batch of predictions.

    Parameters
    ----------
    y : numpy.ndarray
        Ground truth / target values with shape ``(n_samples, n_outputs)``.
    yhat : numpy.ndarray
        Model predictions with the same shape as ``y``.

    Returns
    -------
    float
        The mean squared error for the batch.
    """
    j = np.mean((y-yhat).T @ (y - yhat))
    return j

  def gradient(self, y, yhat):
    """Return the derivative of the mean squared error with respect to ``yhat``.

    Parameters
    ----------
    y : numpy.ndarray
        Ground truth / target values with shape ``(n_samples, n_outputs)``.
    yhat : numpy.ndarray
        Model predictions with the same shape as ``y``.

    Returns
    -------
    numpy.ndarray
        ``-2 * (y - yhat)``, with the same shape as ``yhat``.
    """
    dj = -2*(y-yhat)
    return dj