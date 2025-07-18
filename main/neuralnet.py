import numpy as np



def sigmoid(x):
        return 1/ (1+np.exp(-x))


class SimpleNeuralNet:
   def __init__(self, input_size, hidden_size, output_size):
          # Initialise weights and biases
          self.W1 = np.random.rand(input_size, hidden_size)
          self.b1 = np.random.rand(hidden_size)
          self.W2 = np.random.rand(hidden_size, output_size)
          self.b2 = np.random.rand(output_size)

   def forward(self, x):
        self.z1 = np.dot(x, self.W1) + self.b1
        self.a1 = np.tanh(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2= sigmoid(self.z2)
        return self.a2