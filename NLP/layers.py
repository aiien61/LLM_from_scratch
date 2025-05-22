from icecream import ic
from typing import List
import numpy as np

# TODO: refactor to classes - repeat and sum
# Repeat
D, N = 8, 7
layer_1 = np.random.randn(1, D)   # input
layer_2 = np.repeat(layer_1, N, axis=0)  # forward

d_layer_2 = np.random.randn(N, D) # imaginary gradients
d_layer_1 = np.sum(d_layer_2, axis=0, keepdims=True)  # backward

# Sum
D, N = 8, 7

layer_1 = np.random.randn(N, D)  # input
layer_2 = np.sum(layer_1, axis=0, keepdims=True)  # forward

d_layer_2 = np.random.randn(1, D)  # imaginary gradients
d_layer_1 = np.repeat(d_layer_2, axis=0, keepdims=True)  # backward


# TODO: unittesting it
# MatMul
class MatMul:
    def __init__(self, W: np.ndarray):
        self.params: List[np.ndarray] = [W]
        self.grads: List[float] = [np.zeros_like(W)]
        self.x: List[float] = None

    def forward(self, x):
        W, = self.params
        out = np.dot(x, W)
        self.x = x
        return out

    def backward(self, dout):
        W, = self.params
        dx = np.dot(dout, W.T)
        dW = np.dot(self.x.T, dout)
        self.grads[0][...] = dW
        return dx


# Sigmoid
class Sigmoid:
    def __init__(self):
        self.params, self.grads = [], []
        self.out = None

    def forward(self, x):
        out = 1 / (1 + np.exp(-x))
        self.out = out
        return out

    def backward(self, dout):
        dx = dout * (1.0 - self.out) * self.out
        return dx

# Affine
class Affine:
    def __init__(self, W, b):
        self.params = [W, b]
        self.grads = [np.zeros_like(W), np.zeros_like(b)]
        self.x = None

    def forward(self, x):
        W, b = self.params
        out = np.dot(x, W) + b
        self.x = x
        return out

    def backward(self, dout):
        W, b = self.params
        dx = np.dot(dout, W.T)
        dW = np.dot(self.x.T, dout)
        db = np.sum(dout, axis=0)

        self.grads[0][...] = dW
        self.grads[1][...] = db
        return dx