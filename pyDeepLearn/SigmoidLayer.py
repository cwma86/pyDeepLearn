"""Layer applying the logistic (sigmoid) activation function."""
import logging
import numpy as np

from pyDeepLearn.LayerInterface import Layer


class SigmoidLayer(Layer):
  """Layer applying the logistic (sigmoid) activation function.

  Each element is transformed as ``Y = 1 / (1 + exp(-X))``, squashing the output
  into the range ``(0, 1)``. This makes the layer a natural choice for
  producing probabilities and for use with
  :class:`~pyDeepLearn.LogLoss.LogLoss`.

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
    """Apply the sigmoid activation to ``dataIn``.

    Parameters
    ----------
    dataIn : numpy.ndarray
        Input data with shape ``(n_samples, n_features)``. A 1-D array is
        promoted to a single-row 2-D array.

    Returns
    -------
    numpy.ndarray
        ``1 / (1 + exp(-dataIn))`` computed element-wise, with the same shape
        as ``dataIn``.

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
    Y = 1/(1+np.exp(-dataIn))
    self.setPrevOut(Y)
    return Y

  def gradient(self):
    """Return the per-observation Jacobian of the sigmoid activation.

    Returns
    -------
    numpy.ndarray
        Array with shape ``(n_samples, n_features, n_features)`` whose diagonal
        entries are ``y * (1 - y)``, where ``y`` is the corresponding output.
    """
    dj = np.zeros((self.getPrevOut().shape[0],
                   self.getPrevOut().shape[1],
                   self.getPrevOut().shape[1]))
    for k in range(self.getPrevOut().shape[0]):
      for i in range(self.getPrevOut().shape[1]):
        for j in range(self.getPrevOut().shape[1]):
          if i == j:
            dj[k][i][j] = self.getPrevOut()[k][j] * (1 - self.getPrevOut()[k][j])

    return dj
