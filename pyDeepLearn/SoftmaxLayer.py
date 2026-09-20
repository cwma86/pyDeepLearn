"""Layer applying the softmax activation function."""
import logging
import math
import numpy as np
from pyDeepLearn.LayerInterface import Layer


class SoftmaxLayer(Layer):
  """Layer applying the softmax activation function.

  Each row of the input is exponentiated and normalized so that it sums to one,
  turning a vector of real-valued scores into a probability distribution over
  ``(0, 1)``. It is typically the final layer before a
  :class:`~pyDeepLearn.CrossEntropy.CrossEntropy` objective for multi-class
  classification.

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
    """Apply the softmax activation to each row of ``dataIn``.

    Parameters
    ----------
    dataIn : numpy.ndarray
        Input data with shape ``(n_samples, n_features)``. A 1-D array is
        promoted to a single-row 2-D array.

    Returns
    -------
    numpy.ndarray
        A probability distribution for each row, with the same shape as
        ``dataIn``. Each row sums to one.

    Raises
    ------
    TypeError
        If ``dataIn`` has more than two dimensions.
    """
    # Input data validation checks
    if dataIn.ndim == 1:
      dataIn = np.array([dataIn])
    if dataIn.ndim > 2:
      logging.error(f"invalid input data matrix dimensions {dataIn.ndim}")
      raise TypeError
    self.setPrevIn(dataIn)

    # Calc the exp value (e^x) for each value in the input matrix
    exp_mat = np.zeros(dataIn.shape)
    for i in range(dataIn.shape[0]):  # each row
      for j in range(dataIn.shape[1]):  # each value in the row
        exp_mat[i][j] = math.exp(dataIn[i][j])

    # Create a matrix of each rows summed value
    exp_sums = np.sum(exp_mat, axis=1)

    # Calculate the soft max values
    output_mat = np.zeros(exp_mat.shape)
    for i in range(exp_mat.shape[0]):
      for j in range(exp_mat.shape[1]):
        output_mat[i][j] = exp_mat[i][j]/exp_sums[i]
    self.setPrevOut(output_mat)
    return output_mat

  def gradient(self):
    """Return the per-observation Jacobian of the softmax activation.

    Returns
    -------
    numpy.ndarray
        Array with shape ``(n_samples, n_features, n_features)``. Diagonal
        entries are ``y_i * (1 - y_i)`` and off-diagonal entries are
        ``-y_i * y_j``, where ``y`` is the corresponding softmax output row.
    """
    dk = np.zeros((self.getPrevOut().shape[0],
                   self.getPrevOut().shape[1],
                   self.getPrevOut().shape[1]))
    for k in range(self.getPrevOut().shape[0]):
      for i in range(self.getPrevOut().shape[1]):
        for j in range(self.getPrevOut().shape[1]):
          if i == j:
            # calc diag
            dk[k][i][j] = self.getPrevOut()[k][j]*(1-self.getPrevOut()[k][j])
          else:
            # calc off diag
            dk[k][i][j] = -1 * self.getPrevOut()[k][i] * self.getPrevOut()[k][j]
            dk[k][j][i] = dk[k][i][j]
    return dk
