from pyDeepLearn.LayerInterface import Layer
import logging
import math
import numpy as np

class InputLayer(Layer):
  def __init__(self, dataIn):
    self.meanX = np.mean(dataIn,axis=0)
    # Note using DDOF = 1
    self.stdX = np.std(dataIn,axis=0,ddof=1)
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
    self.setPrevIn(dataIn)

    zscore = (dataIn - self.meanX) / self.stdX
    self.setPrevOut(zscore)
    return zscore

  def gradient(self):
    #TODO not yet implemented
    pass