import numpy as np
from rich import print
from icecream import ic
from typing import List

x: np.ndarray = np.array([1, 2, 3])
ic(x)

ic(x.__class__)  # numpy.ndarray

ic(x.shape)  # (3,)

ic(x.ndim)  # 1

W: np.ndarray = np.array([[1, 2, 3], [4, 5, 6]])
ic(W.__class__)  # numpy.ndarray

ic(W.shape)  # (2, 3)

ic(W.ndim)  # 2

Q: np.ndarray = np.array(
    [
        [[1, 2, 3], [4, 5, 6]], 
        [[7, 8, 9], [10, 11, 12]]
    ]
)
ic(Q.__class__)  # numpy.ndarray

ic(Q.shape)  # (2, 2, 3)

ic(Q.ndim)  # 3

# element-wise operations
W: np.ndarray = np.array([[1, 2, 3], [4, 5, 6]])
X: np.ndarray = np.array([[0, 1, 2], [3, 4, 5]])
ic(W + X)

ic(W * X)

# broadcasting
A: np.ndarray = np.array([[1, 2], [3, 4]])
ic(A * 10)

A: np.ndarray = np.array([[1, 2], [3, 4]])
b: np.ndarray = np.array([10, 20])
ic(A * b)

# dot-product
a: np.ndarray = np.array([1, 2, 3])
b: np.ndarray = np.array([4, 5, 6])
ic(a, b)
ic(np.dot(a, b))

# matrix multiplication
A: np.ndarray = np.array([[1, 2], [3, 4]])
B: np.ndarray = np.array([[5, 6], [7, 8]])
ic(a, b)
ic(np.dot(A, B))

# demo how to connect layers in a neural network

# 2 neurons in the input layer
# 4 hidden neurons in the hidden layer
# 3 neurons in the outpyt layer

x = np.random.randn(10, 2)  # input: 10 sets, and each set has 2 dimensions
W1 = np.random.randn(2, 4)  # weights
b1 = np.random.randn(4)     # bias
W2 = np.random.randn(4, 3)  # weights
b2 = np.random.randn(3)     # bias
ic(x)
ic(W1, W2)
ic(b1, b2)

# activation function to convert number to be output in between 0 and 1
def sigmoid(x):
    return 1 / (1 + np.exp(-x))    

h = np.dot(x, W1) + b1
a = sigmoid(h)
s = np.dot(a, W2) + b2
ic(s)

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

    
# input
x = np.random.randn(10, 2)
ic(x)

model = TwoLayerNet(2, 4, 3)
ic(model)

s = model.predict(x)
ic(s)