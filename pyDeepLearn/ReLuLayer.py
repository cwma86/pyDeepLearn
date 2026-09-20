"""Layer applying the rectified linear (ReLU) activation function."""
from pyDeepLearn.LayerInterface import Layer
import logging
import numpy as np

class ReLuLayer(Layer):
  """Layer applying the rectified linear (ReLU) activation function.

  Each element is transformed as ``Y = max(0, X)``, clamping every negative
  value to zero while passing positive values through unchanged.

  Attributes
  ----------
  __prevIn : numpy.ndarray
      Input data supplied during the most recent forward pass.
  __prevOut : numpy.ndarray
      Output data produced during the most recent forward pass.
  """
  def __init__(self, dataIn):
    """Initialize the layer.

    Parameters
    ----------
    dataIn : numpy.ndarray
        Accepted for interface consistency with the other activation layers;
        the value is not used.
    """
    super().__init__()


  def forward(self, dataIn):
    """Apply the ReLU activation to ``dataIn``.

    Parameters
    ----------
    dataIn : numpy.ndarray
        Input data with shape ``(n_samples, n_features)``. A 1-D array is
        promoted to a single-row 2-D array.

    Returns
    -------
    numpy.ndarray
        ``max(0, dataIn)`` computed element-wise, with the same shape as
        ``dataIn``.

    Raises
    ------
    TypeError
        If ``dataIn`` has more than two dimensions.
    """
    if dataIn.ndim == 1:
      dataIn = np.array([dataIn])
    if dataIn.ndim > 2:
      logging.error(f"invalid input data matrix dimensions {dataIn.ndim }")
      raise TypeError

    self.setPrevIn(dataIn)
    # Initialize a equal size matrix of all 0 values
    zero_matrix = np.zeros(dataIn.shape)
    # Replace any negative with 0
    Y = np.maximum(zero_matrix, dataIn)
    self.setPrevOut(Y)
    return Y

  def gradient(self):
    """Return the per-observation Jacobian of the ReLU activation.

    Returns
    -------
    numpy.ndarray
        Array with shape ``(n_samples, n_features, n_features)`` whose diagonal
        entries are ``1`` where the corresponding input was non-negative and
        ``0`` where it was negative.
    """
    dj = np.zeros((self.getPrevOut().shape[0],
                self.getPrevOut().shape[1],
                self.getPrevOut().shape[1]))
    for k in range(self.getPrevOut().shape[0]):
      for i in range(self.getPrevOut().shape[1]):
        for j in range(self.getPrevOut().shape[1]):
          if i == j:
            if self.getPrevIn()[k][j] < 0 :
              dj[k][i][j] = 0
            else :
              dj[k][i][j] = 1

    return dj