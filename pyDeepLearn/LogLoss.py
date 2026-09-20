"""Binary log loss (binary cross entropy) objective function."""
import numpy as np
from pyDeepLearn.objectiveFuncInterface import objectiveFuncInterface


class LogLoss(objectiveFuncInterface):
  """Objective function computing binary cross entropy (log loss).

  The loss for a batch is the mean over the batch of

  ``-sum(y * log(yhat + E) + (1 - y) * log(1 - yhat + E))``

  where ``E`` is a small constant (``1e-8``) that keeps the logarithms finite.
  The objective is intended for a binary target produced by a
  :class:`~pyDeepLearn.SigmoidLayer.SigmoidLayer`.
  """

  def eval(self, y, yhat):
    """Evaluate the mean binary cross entropy for a batch of predictions.

    Parameters
    ----------
    y : numpy.ndarray
        Binary ground truth / target values with shape
        ``(n_samples, n_outputs)``.
    yhat : numpy.ndarray
        Predicted probabilities with the same shape as ``y``. Values should lie
        within ``(0, 1)``.

    Returns
    -------
    float
        The mean binary cross entropy for the batch.
    """
    E = 1e-8
    j = np.zeros(y.shape[0])
    for i in range(y.shape[0]):
      j[i] = np.sum(-((y[i]*np.log(yhat[i] + E) + (1-y[i]) * np.log(1-yhat[i] + E))), 0)/y[i].shape[0]
    j = np.mean(j)
    return j

  def gradient(self, y, yhat):
    """Return the derivative of the binary cross entropy with respect to ``yhat``.

    Parameters
    ----------
    y : numpy.ndarray
        Binary ground truth / target values with shape
        ``(n_samples, n_outputs)``.
    yhat : numpy.ndarray
        Predicted probabilities with the same shape as ``y``.

    Returns
    -------
    numpy.ndarray
        ``-(y - yhat) / (yhat * (1 - yhat) + E)``, with the same shape as
        ``yhat``.
    """
    E = 1e-8
    dj = -1 * (y - yhat) / (yhat*(1-yhat) + E)
    return dj
