"""Cross entropy objective function for multi-class classification."""
import logging
import math
import numpy as np
from pyDeepLearn.objectiveFuncInterface import objectiveFuncInterface

class CrossEntropy(objectiveFuncInterface):
  """Objective function computing cross entropy for multi-class classification.

  The loss for a batch is the mean over the batch of ``-sum(y * log(yhat + E))``
  where ``E`` is a small constant that keeps the logarithm finite. It is
  intended to be paired with a one-hot encoded target and a
  :class:`~pyDeepLearn.SoftmaxLayer.SoftmaxLayer` output.
  """
  def eval(self, y, yhat):
    """Evaluate the mean cross entropy for a batch of predictions.

    Parameters
    ----------
    y : numpy.ndarray
        One-hot encoded ground truth / target values with shape
        ``(n_samples, n_classes)``. A 1-D array is promoted to a single-row
        2-D array.
    yhat : numpy.ndarray
        Model predictions (typically softmax probabilities) with the same shape
        as ``y``.

    Returns
    -------
    float
        The mean cross entropy for the batch.

    Raises
    ------
    TypeError
        If ``y`` has more than two dimensions.
    """
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
    """Return the derivative of the cross entropy with respect to ``yhat``.

    Parameters
    ----------
    y : numpy.ndarray
        One-hot encoded ground truth / target values with shape
        ``(n_samples, n_classes)``.
    yhat : numpy.ndarray
        Model predictions with the same shape as ``y``.

    Returns
    -------
    numpy.ndarray
        ``-y / (yhat + E)``, with the same shape as ``yhat``.
    """
    E = 0.00000001
    dj = -1*(y/(yhat+E))
    return dj