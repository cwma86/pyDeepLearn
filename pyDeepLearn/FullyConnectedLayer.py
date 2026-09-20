"""Fully connected (dense) layer and its recurrent variant.

``FullyConnectedLayer`` applies an affine transform and learns a weight matrix
and bias. ``RecurrentFcLayer`` specializes it for use with the recurrent
training helpers in :mod:`pyDeepLearn.RunUtilities`.
"""
import logging
import math
import numpy as np
import sys

from pyDeepLearn.LayerInterface import Layer

class FullyConnectedLayer(Layer):
  """A fully connected (dense) layer that changes data dimensionality.

  The layer applies the affine transform ``h = dataIn @ weights + bias``, which
  can expand or reduce the number of features. Weights and bias are updated by
  the training loop, either with gradient descent (:meth:`updateWeights`), the
  recurrent update used by the RNN helpers
  (:meth:`reccurentWeightUpdate`), or the Adam optimizer
  (:meth:`adam_weight_update`).
  """
  def __init__(self, sizeIn, sizeOut, 
               weight=None, bias=None, 
               weight_up_func="updateWeights",
               eta=0.001):
    """Initialize the layer's weights, bias, and optimizer settings.

    Parameters
    ----------
    sizeIn : int
        Number of input features (number of rows in the weight matrix).
    sizeOut : int
        Number of output features (number of columns in the weight matrix).
    weight : list of float, optional
        Two-element ``[low, high]`` bounds passed to
        :func:`numpy.random.uniform` to initialize the weights. When ``None``
        the weights are drawn uniformly from ``[-0.0001, 0.0001]``.
    bias : list of float, optional
        Two-element ``[low, high]`` bounds used to initialize the bias. When
        ``None`` the bias is drawn uniformly from ``[-0.0001, 0.0001]``.
    weight_up_func : str, optional
        Weight update strategy. Pass ``"adam_weight_update"`` to enable the
        Adam optimizer; any other value (the default) uses
        :meth:`updateWeights`.
    eta : float, optional
        Learning rate used by the weight update. Defaults to ``0.001``.

    Attributes
    ----------
    weights : numpy.ndarray
        Weight matrix with shape ``(sizeIn, sizeOut)``.
    bias : numpy.ndarray
        Bias vector with shape ``(sizeOut,)``.
    eta : float
        Learning rate.
    weight_up_func : callable
        Bound method used to apply a weight update.
    epoch : int
        Counter used to schedule weight decay/jitter during training.
    """
    super().__init__()

    w_size = [sizeIn, sizeOut]
    self.setPrevIn(np.zeros(sizeIn))
    self.setPrevOut(np.zeros((1,sizeOut)))
    self.eta = eta
    if weight == None:
      self.weights = np.random.uniform(-0.0001, 0.0001, size=w_size)
    else:
      self.weights = np.random.uniform(weight[0], weight[1], size=w_size)
    if bias == None:
      self.bias = np.random.uniform(-0.0001, 0.0001, size=sizeOut)
    else:
      self.bias = np.random.uniform(bias[0], bias[1], size=sizeOut)

    if weight_up_func == "adam_weight_update":
      self.s = 0
      self.r = 0
      self.p1=0.9
      self.p2=0.999
      self.sig=10e-8
      self.weight_up_func = self.adam_weight_update
    else:
      self.weight_up_func = self.updateWeights
    self.epoch = 0 

  def getWeights(self):
    """Return the layer's weight matrix.

    Returns
    -------
    numpy.ndarray
        Weight matrix with shape ``(sizeIn, sizeOut)``.
    """
    return self.weights

  def setWeights(self, weights):
    """Replace the layer's weight matrix.

    Parameters
    ----------
    weights : numpy.ndarray
        New weight matrix with shape ``(sizeIn, sizeOut)``.
    """
    self.weights = weights
  
  def getBias(self):
    """Return the layer's bias vector.

    Returns
    -------
    numpy.ndarray
        Bias vector with shape ``(sizeOut,)``.
    """
    return self.bias
  
  def setBias(self, bias):
    """Replace the layer's bias vector.

    Parameters
    ----------
    bias : numpy.ndarray
        New bias vector with shape ``(sizeOut,)``.
    """
    self.bias = bias

  def forward(self, dataIn):
    """Apply the affine transform ``dataIn @ weights + bias``.

    Parameters
    ----------
    dataIn : numpy.ndarray
        Input data with shape ``(n_samples, sizeIn)``. A 1-D array is promoted
        to a single-row 2-D array.

    Returns
    -------
    numpy.ndarray
        Transformed data with shape ``(n_samples, sizeOut)``.

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
    # print(f" dataIn { dataIn.shape}")
    # print(f" self.weights { self.weights.shape}")
    # print(f" self.bias { self.bias.shape}")
    try:
      h = dataIn @ self.weights + self.bias
    except RuntimeWarning:
      logging.info(f"Warn!")
    # print(f" h  { h .shape}")
    self.setPrevOut(h)
    return h

  def gradient(self):
    """Return the layer's local Jacobian for each observation.

    Returns
    -------
    numpy.ndarray
        Array with shape ``(n_samples, sizeIn, sizeOut)`` in which every
        per-observation matrix is the transpose of the weight matrix.
    """
    dj = []
    for i in range(len(self.getPrevIn())):
      dj.append(self.getWeights().T)
    dj = np.array(dj)
    array_sum = np.sum(dj)
    if np.isnan(array_sum):
      logging.error("gradient is nan")
      print(len(self.getPrevIn()))
      print(self.getWeights())
      sys.exit(1)
    return dj
  
  def updateWeights(self, gradIn, epoch=1):
    """Apply a gradient-descent update to the weights and bias.

    Parameters
    ----------
    gradIn : numpy.ndarray
        Gradient of the objective with respect to the layer output, with shape
        ``(n_samples, sizeOut)``.
    epoch : int, optional
        Current training epoch. Every 10 epochs the weights are multiplied by
        ``0.999`` to add a small amount of decay/jitter. Defaults to ``1``.

    Notes
    -----
    The weights are updated with ``dJ/dW = prevIn.T @ gradIn / n_samples`` and
    the bias with ``dJ/db = sum(gradIn, axis=0) / n_samples``.
    """
    dJdw= (self.getPrevIn().T @ gradIn)/gradIn.shape[0]
    self.weights = self.weights - self.eta * dJdw
    
    # add jitter after so many epochs
    if epoch % 10 == 0:
      self.weights = self.weights * 0.999
      
    dJdb = np.sum(gradIn, axis = 0)/gradIn.shape[0]
    self.bias = self.bias - self.eta * dJdb

  def reccurentWeightUpdate(self, djdw, djdb):
    """Apply a weight and bias update from pre-accumulated RNN gradients.

    Parameters
    ----------
    djdw : numpy.ndarray
        Accumulated gradient of the objective with respect to the weights.
    djdb : numpy.ndarray
        Accumulated gradient of the objective with respect to the bias.

    Notes
    -----
    Because the recurrent training helpers accumulate gradients across time
    steps before calling this method, the accumulated values are applied
    directly rather than being divided by the batch size. Every 750 calls the
    weights are multiplied by ``0.999`` to add a small amount of jitter.
    """
    array_sum = np.sum(djdw)
    if np.isnan(array_sum):
      logging.error("gradient is nan")
      sys.exit(1)
    # add jitter after so many epochs
    self.weights = self.weights - self.eta * djdw
    if self.epoch % 750 == 0:
      self.weights = self.weights * 0.999
      self.epoch = 1
    self.epoch += 1


    # TODO not really sure what to do with the bias here...
    self.bias = self.bias - self.eta * djdb

  def adam_weight_update(self, gradIn, epoch=1):
    """Apply an Adam optimizer update to the weights and bias.

    First- and second-moment estimates (``s`` and ``r``) are maintained using
    the decay rates ``p1 = 0.9`` and ``p2 = 0.999`` and are bias-corrected
    before the weights are updated. The bias is updated with plain gradient
    descent.

    Parameters
    ----------
    gradIn : numpy.ndarray
        Gradient of the objective with respect to the layer output, with shape
        ``(n_samples, sizeOut)``.
    epoch : int, optional
        Current training epoch, used for the bias-correction terms. Defaults
        to ``1``.
    """
    self.s = self.p1 * self.s + ((1-self.p1) * np.sum( gradIn, axis=0)/gradIn.shape[0])
    self.r = self.p2 * self.r + ((1-self.p2) * np.sum((gradIn * gradIn), axis=0)/gradIn.shape[0])

    temp_s = self.s/(1-self.p1**epoch)
    temp_r = self.r/(1-self.p2**epoch)
    self.weights = self.weights - self.eta *(temp_s/(np.sqrt(temp_r)+self.sig))
    dJdb = np.sum(gradIn, axis = 0)/gradIn.shape[0]
    self.bias = self.bias - self.eta * dJdb

class RecurrentFcLayer(FullyConnectedLayer):
  """A fully connected layer with recurrent (across-time) state.

  ``RecurrentFcLayer`` shares the weights, bias, and update rules of
  :class:`FullyConnectedLayer`, but the RNN helpers in
  :mod:`pyDeepLearn.RunUtilities` treat it specially: its previous output is
  carried forward between time steps and summed into the next forward pass, and
  its weights are updated from gradients accumulated across the sequence via
  :meth:`FullyConnectedLayer.reccurentWeightUpdate`.
  """
  def __init__(self, sizeIn, sizeOut, 
               weight=None, bias=None, 
               weight_up_func="updateWeights",
               eta=0.001):
    """Initialize the recurrent fully connected layer.

    Parameters
    ----------
    sizeIn : int
        Number of input features.
    sizeOut : int
        Number of output features.
    weight : list of float, optional
        Two-element ``[low, high]`` bounds used to initialize the weights.
    bias : list of float, optional
        Two-element ``[low, high]`` bounds used to initialize the bias.
    weight_up_func : str, optional
        Weight update strategy; see :class:`FullyConnectedLayer`.
    eta : float, optional
        Learning rate. Defaults to ``0.001``.
    """
    super().__init__( sizeIn=sizeIn, sizeOut=sizeOut, 
               weight=weight, bias=bias, 
               weight_up_func=weight_up_func,
               eta=eta)
