import numpy as np
from rich import print
from icecream import ic
from typing import List

# demo mini-batch layers connection in a neural network

# 2 neurons in the input layer
# 4 hidden neurons in the hidden layer
# 3 neurons in the outpyt layer
# 10 input sets. Each set has 2 features

x: np.ndarray = np.random.randn(10, 2)  # input: 10 sets, and each set has 2 features
W1: np.ndarray = np.random.randn(2, 4)  # weights
b1: np.ndarray = np.random.randn(4)     # bias
W2: np.ndarray = np.random.randn(4, 3)
b2: np.ndarray = np.random.randn(3)     # bias


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


h: np.ndarray = np.dot(x, W1) + b1
a: np.ndarray = sigmoid(h)
s: np.ndarray = np.dot(a, W2) + b2

ic(W1)
ic(W1.shape)
ic(W1.size)
ic(W1.ndim)

ic(b1)
ic(b1.shape)
ic(b1.size)
ic(b1.ndim)

ic(x)
ic(x.shape)
ic(x.size)
ic(x.ndim)

ic(h)
ic(h.shape)
ic(h.size)
ic(h.ndim)

ic(W2)
ic(W2.shape)
ic(W2.size)
ic(W2.ndim)

ic(b2)
ic(b2.shape)
ic(b2.size)
ic(b2.ndim)

ic(a)
ic(a.shape)
ic(a.size)
ic(a.ndim)

ic(s)
ic(s.shape)
ic(s.size)
ic(s.ndim)


# x = np.random.randn(10, 2)  # input: 10 sets, and each set has 2 dimensions
# W1 = np.random.randn(2, 4)  # weights
# b1 = np.random.randn(4)     # bias
# W2 = np.random.randn(4, 3)  # weights
# b2 = np.random.randn(3)     # bias
# ic(x)
# ic(W1, W2)
# ic(b1, b2)

# activation function to convert number to be output in between 0 and 1
def sigmoid(x):
    return 1 / (1 + np.exp(-x))    

# h = np.dot(x, W1) + b1
# a = sigmoid(h)
# s = np.dot(a, W2) + b2
# ic(s)

# building a simple 2-layer neural network
class Sigmoid:
    def __init__(self):
        self.params = []

    def forward(self, x):
        return 1 / (1 + np.exp(-x))
    
    def __str__(self) -> str:
        return f"Sigmoid(params={self.params})"
    
    def __repr__(self) -> str:
        return str(self)

    
class Affine:
    def __init__(self, W, b):
        self.params = [W, b]
    
    def forward(self, x):
        W, b = self.params
        out = np.dot(x, W) + b
        return out
    
    def __str__(self) -> str:
        return f"Affine(params={self.params})"
    
    def __repr__(self) -> str:
        return str(self)


class TwoLayerNet:
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        I, H, O = input_size, hidden_size, output_size

        # initialises weights and bias
        W1 = np.random.randn(I, H)  
        b1 = np.random.randn(H)     
        W2 = np.random.randn(H, O)  
        b2 = np.random.randn(O)

        # generates each layer
        self.layers = [
            Affine(W1, b1),
            Sigmoid(),
            Affine(W2, b2)
        ]

        # stores all the weights in a list
        self.params = []
        for layer in self.layers:
            self.params += layer.params

    def predict(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x
    
    def __str__(self) -> str:
        layers: List[str] = [str(layer) for layer in self.layers]
        return '\n-> '.join(layers)

    def __repr__(self) -> str:
        return str(self)

    
# # input
# x = np.random.randn(10, 2)
# ic(x)

# model = TwoLayerNet(2, 4, 3)
# ic(model)

# s = model.predict(x)
# ic(s)