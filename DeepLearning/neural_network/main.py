import numpy as np
from icecream import ic
from utils import n
from nn_module import MyNet
from rich import print
from activation_function import sigmoid, identity_function

def main(*args):
    # Inputs
    X: np.ndarray = n((2,), *(1, 0.5))
    ic(X)
    ic(X.shape)

    # Layer 1
    W1: np.ndarray = n((2, 3), *(0.1, 0.3, 0.5, 0.2, 0.4, 0.6))
    ic(W1)
    ic(W1.shape)

    B1: np.ndarray = n((3,), *(0.1, 0.2, 0.3))
    ic(B1)
    ic(B1.shape)

    A1: np.ndarray = np.dot(X, W1) + B1
    ic(A1)
    ic(A1.shape)

    Z1: np.ndarray = sigmoid(A1)
    ic(Z1)
    ic(Z1.shape)

    # Layer 2
    W2: np.ndarray = n((3, 2), *(0.1, 0.4, 0.2, 0.5, 0.3, 0.6))
    ic(W2)
    ic(W2.shape)

    B2: np.ndarray = n((2,), *(0.1, 0.2))
    ic(B2)
    ic(B2.shape)

    A2: np.ndarray = np.dot(Z1, W2) + B2
    ic(A2)
    ic(A2.shape)

    Z2: np.ndarray = sigmoid(A2)
    ic(Z2)
    ic(Z2.shape)

    # Layer 3
    W3: np.ndarray = n((2, 2), *(0.1, 0.3, 0.2, 0.4))
    ic(W3)
    ic(W3.shape)

    B3: np.ndarray = n((2,), *(0.1, 0.2))
    ic(B3)
    ic(B3.shape)

    A3: np.ndarray = np.dot(Z2, W3) + B3
    ic(A3)
    ic(A3.shape)

    Z3: np.ndarray = identity_function(A3)
    ic(Z3)
    ic(Z3.shape)

if __name__ == "__main__":
    network = MyNet.init_network()
    x = np.array([1.0, 0.5])
    y = MyNet.forward(network, x)
    print(y)