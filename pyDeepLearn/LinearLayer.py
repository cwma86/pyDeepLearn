"""Layer that passes its input through unchanged (the identity transform)."""
import logging
import numpy as np

from pyDeepLearn.LayerInterface import Layer


class LinearLayer(Layer):
  """Layer that passes its input through unchanged.

  ``LinearLayer`` performs the identity transformation ``Y = X``. It is useful
  as a gradient pass-through (its Jacobian is the identity matrix), which makes
  it convenient to place between a fully connected layer and an objective
  function.

  Attributes
  ----------
  __prevIn : numpy.ndarray
      Input data supplied during the most recent forward pass.
  __prevOut : numpy.ndarray
      Output data produced during the most recent forward pass.
  """

  def __init__(self):
    """Initialize the layer. The layer has no tunable parameters."""
    super().__init__()

  def forward(self, dataIn):
    """Return ``dataIn`` unchanged, caching it as the previous input/output.

    Parameters
    ----------
    dataIn : numpy.ndarray
        Input data with shape ``(n_samples, n_features)``. A 1-D array is
        promoted to a single-row 2-D array.

    Returns
    -------
    numpy.ndarray
        The unchanged input data.

    Raises
    ------
    TypeError
        If ``dataIn`` has more than two dimensions.
    """
    # Input data validation checks
    if dataIn.ndim == 1:
      dataIn = np.array([dataIn])
    if dataIn.ndim > 2:
      logging.error(f"invalid input data matrix dimensions {dataIn.ndim }")
      raise TypeError
    self.setPrevIn(dataIn)
    Y = dataIn
    self.setPrevOut(Y)
    return Y

  def gradient(self):
    """Return the identity Jacobian for each observation in the batch.

    Returns
    -------
    numpy.ndarray
        Array with shape ``(n_samples, n_features, n_features)`` whose
        per-observation matrices are the identity matrix.
    """
    dj = np.zeros((self.getPrevOut().shape[0],
                   self.getPrevOut().shape[1],
                   self.getPrevOut().shape[1]))
    for k in range(self.getPrevOut().shape[0]):
      for i in range(self.getPrevOut().shape[1]):
        for j in range(self.getPrevOut().shape[1]):
          if i == j:
            dj[k][i][j] = 1
    return dj
