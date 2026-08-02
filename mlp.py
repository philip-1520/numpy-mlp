import numpy as np
rng = np.random.default_rng()

class MultilayerPerceptron:

  def __init__(self, layers: list, learning_rate: float, loss_function) -> None:

    self.layers = layers
    self.learning_rate = learning_rate
    self.loss_function = loss_function
    self.size = len(self.layers)

    for layer_index in range(1, self.size):

      layer = self.layers[layer_index]
      previous_layer = self.layers[layer_index-1]

      layer.x = np.zeros(previous_layer.size)
      layer.W = rng.standard_normal((previous_layer.size, layer.size)) * np.sqrt(2/layer.size)
      layer.b = np.zeros(layer.size)
      layer.z = np.zeros(layer.size)

  def input(self, entry) -> None:
    self.layers[0].y = entry

  @property
  def output(self):
    return self.layers[-1].y

  @property
  def summary(self):
    architecture = dict()
    architecture[0] = {'y': self.layers[0].y}
    for layer_index in range(1, self.size):
      layer = self.layers[layer_index]
      architecture[layer_index] = {
          'x': layer.x,
          'W': layer.W,
          'b': layer.b,
          'z': layer.z,
          'y': layer.y
          }

    return architecture

  def feedforward(self) -> None:
    for layer_index in range(1, self.size):
      layer = self.layers[layer_index]
      previous_layer = self.layers[layer_index-1]
      layer.x = previous_layer.y
      layer.compute()

  def backpropagate(self, target) -> None:
    grad_signal = self.loss_function.derivative(self.output, target)
    grad_signal = self.learning_rate * grad_signal

    #Iteração sobre todas as camadas ocultas
    for layer in self.layers[-1:0:-1]:
      grad_signal = layer.learn(grad_signal)
    
  def train(self, x_train, y_train, epochs):

    training_history = dict()
    training_history['loss'] = dict()
    training_history['correct'] = dict()

    for epoch in range(1, epochs + 1):
      epoch_loss = 0
      epoch_correct = 0

      for entry, target in zip(x_train, y_train):

        self.input(np.array(entry))
        self.feedforward()

        epoch_loss += self.loss_function.function(self.output, target)
        if np.argmax(self.output) == target:
          epoch_correct += 1

        self.backpropagate(target)

      training_history['loss'][epoch] = epoch_loss
      training_history['correct'][epoch] = epoch_correct

    return training_history
    
  def predict(self, entry_array):
    prediction = list()
    for entry in entry_array:
      self.input(np.array(entry))
      self.feedforward()
      prediction.append(np.argmax(self.output))
    return prediction

class Layer:

  def __init__(self, size, activation_function=None) -> None:

    self.size = size
    self.activation_function = activation_function

    self.x = None
    self.W = None
    self.b = None
    self.z = None
    self.y = np.zeros(size)

  def compute(self) -> None:
    self.z = (self.x @ self.W) + self.b
    self.y = self.activation_function.function(self.z)

  def learn(self, grad_signal):
    delta = grad_signal @ self.activation_function.derivative(self.z)

    grad_w = np.outer(self.x, delta)
    grad_b = delta

    untrained_weight = self.W
    self.W -= grad_w
    self.b -= grad_b

    return untrained_weight @ delta
    
