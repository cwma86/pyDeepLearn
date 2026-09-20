"""Abstract interface shared by every objective (loss) function.

An objective function measures how far a model's predictions are from the
ground truth and supplies the gradient of that measurement with respect to the
predictions. Every objective function in :mod:`pyDeepLearn` (for example
:class:`~pyDeepLearn.LeastSquares.LeastSquares` or
:class:`~pyDeepLearn.CrossEntropy.CrossEntropy`) subclasses
:class:`objectiveFuncInterface`.
"""
from abc import ABC, abstractmethod
import numpy as np
class objectiveFuncInterface(ABC):
  """Abstract base class for a deep learning objective (loss) function.

  Subclasses must implement :meth:`eval`, which scores a batch of predictions,
  and :meth:`gradient`, which returns the derivative of that score with respect
  to the predictions. The gradient produced by :meth:`gradient` is the value
  that seeds back propagation through the network.
  """
  @abstractmethod
  def eval(self, y, yhat):
    """Evaluate the objective (loss) for a batch of predictions.

    Parameters
    ----------
    y : numpy.ndarray
        Ground truth / target values with shape ``(n_samples, n_outputs)``.
    yhat : numpy.ndarray
        Model predictions with the same shape as ``y``.

    Returns
    -------
    float
        The scalar objective value for the batch.
    """
    pass
  @abstractmethod
  def gradient(self, y, yhat):
    """Compute the gradient of the objective with respect to the predictions.

    Parameters
    ----------
    y : numpy.ndarray
        Ground truth / target values with shape ``(n_samples, n_outputs)``.
    yhat : numpy.ndarray
        Model predictions with the same shape as ``y``.

    Returns
    -------
    numpy.ndarray
        Derivative of the objective with respect to ``yhat``, with the same
        shape as ``yhat``.
    """
    pass

