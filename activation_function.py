import numpy as np

class ReLU:

  @staticmethod
  def function(z):
    return np.where(z >= 0, z, 0)

  @staticmethod
  def derivative(z):
    d = np.identity(len(z))
    np.fill_diagonal(d, max(0, 1))
    return d

class Softmax:

  @staticmethod
  def function(z):
    z = z - np.max(z)
    return np.exp(z)/np.sum(np.exp(z))

  @staticmethod
  def derivative(z):
    y = Softmax.function(z)
    d = np.outer(y, -y)
    np.fill_diagonal(d, y*(1-y))
    return d
