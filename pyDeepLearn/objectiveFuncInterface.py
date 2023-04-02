from abc import ABC, abstractmethod
import numpy as np
class objectiveFuncInterface(ABC):
  @abstractmethod
  def eval(self, y, yhat):
    pass
  @abstractmethod
  def gradient(self, y, yhat):
    pass
