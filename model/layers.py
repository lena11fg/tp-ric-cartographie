import numpy as np

class Dense:
    def __init__(self, input_size, output_size):
        # Initialisation Xavier 
        self.W = np.random.randn(input_size, output_size) * np.sqrt(1 / input_size)
        # self.W = np.random.randn(input_size, output_size) * np.sqrt(2 / input_size)
        self.b = np.zeros((1, output_size))

    def forward(self, X):
        self.X = X  # stock pour backprop plus tard
        return X @ self.W + self.b