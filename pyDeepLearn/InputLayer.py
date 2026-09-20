"""Layer that standardizes (z-score normalizes) its input data.

The per-feature mean and standard deviation of the training data are captured
at construction time and reused for every subsequent forward pass.
"""
from pyDeepLearn.LayerInterface import Layer
import logging
import math
import numpy as np


class InputLayer(Layer):
  """Layer that standardizes (z-score normalizes) its input data.

  The layer records the per-feature mean and standard deviation of the data
  supplied at construction time. :meth:`forward` then standardizes each feature
  as ``(x - mean) / std`` so that every feature shares a comparable scale. A
  feature whose standard deviation is zero is given a standard deviation of
  ``1.0`` to avoid division by zero.

  Attributes
  ----------
  meanX : numpy.ndarray
      Per-feature mean of the construction data.
  stdX : numpy.ndarray
      Per-feature standard deviation (``ddof=1``) of the construction data,
      with any zero values replaced by ``1.0``.

  Notes
  -----
  :meth:`gradient` is not implemented for this layer.
  """

  def __init__(self, dataIn):
    """Capture the per-feature mean and standard deviation of ``dataIn``.

    Parameters
    ----------
    dataIn : numpy.ndarray
        Training data with shape ``(n_samples, n_features)`` used to compute
        the normalization statistics.
    """
    self.meanX = np.mean(dataIn, axis=0)
    # Note using DDOF = 1
    self.stdX = np.std(dataIn, axis=0, ddof=1)
    min_stddev = np.amin(self.stdX)
    if math.isclose(min_stddev, 0.0):
      for i in range(len(self.stdX)):
        if math.isclose(self.stdX[i], 0.0):
          # For stability set std dev of 0 to 1.
          self.stdX[i] = 1.0
    logging.info(f"self.meanX {self.meanX}")
    logging.info(f"self.stdX {self.stdX}")
    super().__init__()

  def forward(self, dataIn):
    """Standardize ``dataIn`` using the stored mean and standard deviation.

    Parameters
    ----------
    dataIn : numpy.ndarray
        Input data with shape ``(n_samples, n_features)``.

    Returns
    -------
    numpy.ndarray
        The z-score normalized data, with the same shape as ``dataIn``.
    """
    self.setPrevIn(dataIn)

    zscore = (dataIn - self.meanX) / self.stdX
    self.setPrevOut(zscore)
    return zscore

  def gradient(self):
    """Return the local gradient of this layer.

    Notes
    -----
    Not yet implemented; returns ``None``.
    """
    # TODO not yet implemented
    pass
