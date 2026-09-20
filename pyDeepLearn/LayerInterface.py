"""Abstract base class for the layers used by the pyDeepLearn framework.

Every layer in :mod:`pyDeepLearn` derives from :class:`Layer`. A layer caches
the input and output of its most recent forward pass so that it can compute its
local gradient and back propagate an incoming gradient without needing a
reference to the rest of the network.
"""
from abc import ABC, abstractmethod
import logging
import numpy as np
import sys


class Layer(ABC):
  """Abstract base class for a single layer in a neural network.

  ``Layer`` is the primary base type of every layer in this deep learning
  module, including input layers, fully connected (dense) layers, recurrent
  layers, and activation (element-wise) layers. It stores the input and output
  of the most recent :meth:`forward` pass so that :meth:`gradient` and
  :meth:`backward` can use them.

  Attributes
  ----------
  __prevIn : numpy.ndarray
      Input data supplied to the layer during the most recent forward pass.
  __prevOut : numpy.ndarray
      Output data produced by the layer during the most recent forward pass.
  """

  def __init__(self):
    """Initialize the cached previous input and output to empty lists."""
    self.__prevIn = []
    self.__prevOut = []

  def setPrevIn(self, dataIn):
    """Store the input data used by the most recent forward pass.

    Parameters
    ----------
    dataIn : numpy.ndarray
        Input data supplied to the layer.
    """
    self.__prevIn = dataIn

  def setPrevOut(self, out):
    """Store the output produced by the most recent forward pass.

    Parameters
    ----------
    out : numpy.ndarray
        Output data produced by the layer.
    """
    self.__prevOut = out

  def getPrevIn(self):
    """Return the input data cached by the most recent forward pass.

    Returns
    -------
    numpy.ndarray
        The previously supplied input data.
    """
    return self.__prevIn

  def getPrevOut(self):
    """Return the output data cached by the most recent forward pass.

    Returns
    -------
    numpy.ndarray
        The previously produced output data.
    """
    return self.__prevOut

  def backward(self, gradIn):
    """Back propagate an incoming gradient through this layer.

    The incoming gradient is multiplied by the layer's local gradient (see
    :meth:`gradient`) for each observation in the batch.

    Parameters
    ----------
    gradIn : numpy.ndarray
        Gradient of the objective with respect to this layer's output, with
        shape ``(n_samples, n_outputs)``.

    Returns
    -------
    numpy.ndarray
        Gradient of the objective with respect to this layer's input, with
        shape ``(n_samples, n_inputs)``.
    """
    sg = self.gradient()
    try:
      grad = np.zeros((gradIn.shape[0], sg.shape[2]))
      for n in range(gradIn.shape[0]):  # compute for each observation in batch
        grad[n, :] = gradIn[n, :]@sg[n, :, :]

    except IndexError:
      logging.error("Invalid shape")
      logging.info(f"gradIn.shape {gradIn.shape} sg.shape {sg.shape}")
      sys.exit(1)
    except RuntimeWarning:
      logging.info("Warn!")
    return grad

  @abstractmethod
  def forward(self, dataIn):
    """Apply the layer's transformation to ``dataIn``.

    Parameters
    ----------
    dataIn : numpy.ndarray
        Input data with shape ``(n_samples, n_inputs)``.

    Returns
    -------
    numpy.ndarray
        Transformed data with shape ``(n_samples, n_outputs)``.
    """
    pass

  @abstractmethod
  def gradient(self, dataIn):
    """Return the layer's local gradient.

    Parameters
    ----------
    dataIn : numpy.ndarray
        Input data used to evaluate the local gradient.

    Returns
    -------
    numpy.ndarray
        Per-observation Jacobian of the layer.
    """
    pass
