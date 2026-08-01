import numpy as np

class SparseCategoricalCrossentropy:

  @staticmethod
  def function(y, target):
    return -np.log(y[target])

  @staticmethod
  def derivative(y, target):
    loss = np.zeros(len(y))
    loss[target] = -1/y[target]
    return loss
