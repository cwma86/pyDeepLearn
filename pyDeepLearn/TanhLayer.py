"""Layer applying the hyperbolic tangent (tanh) activation function."""
import logging
import numpy as np

from pyDeepLearn.LayerInterface import Layer


class TanhLayer(Layer):
  """Layer applying the hyperbolic tangent (tanh) activation function.

  Each element is transformed as
  ``Y = (exp(X) - exp(-X)) / (exp(X) + exp(-X))``, squashing the output into the
  range ``(-1, 1)``.

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
    """Apply the tanh activation to ``dataIn``.

    Parameters
    ----------
    dataIn : numpy.ndarray
        Input data with shape ``(n_samples, n_features)``. A 1-D array is
        promoted to a single-row 2-D array.

    Returns
    -------
    numpy.ndarray
        The element-wise hyperbolic tangent of ``dataIn``, with the same shape
        as ``dataIn``.

    Raises
    ------
    TypeError
        If ``dataIn`` has more than two dimensions.
    """
    if dataIn.ndim == 1:
      dataIn = np.array([dataIn])
    if dataIn.ndim > 2:
      logging.error(f"invalid input data matrix dimensions {dataIn.ndim}")
      raise TypeError
    self.setPrevIn(dataIn)
    Y = (np.exp(dataIn) - np.exp(-dataIn)) / (
          np.exp(dataIn) + np.exp(-dataIn)
        )
    self.setPrevOut(Y)
    return Y

  def gradient(self):
    """Return the per-observation Jacobian of the tanh activation.

    Returns
    -------
    numpy.ndarray
        Array with shape ``(n_samples, n_features, n_features)`` whose diagonal
        entries are ``1 - y**2`` (plus a small epsilon for numerical stability),
        where ``y`` is the corresponding output.
    """
    dj = np.zeros((self.getPrevOut().shape[0],
                   self.getPrevOut().shape[1],
                   self.getPrevOut().shape[1]))
    E = 0.000001
    for k in range(self.getPrevOut().shape[0]):
      for i in range(self.getPrevOut().shape[1]):
        for j in range(self.getPrevOut().shape[1]):
          if i == j:
            dj[k][i][j] = 1 - (self.getPrevOut()[k][j]**2) + E
    return dj
