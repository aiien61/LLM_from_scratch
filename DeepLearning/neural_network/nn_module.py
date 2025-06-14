import numpy as np
from utils import n
from typing import Dict
from activation_function import sigmoid, identity_function


class MyNet:
    def init_network():
        network: Dict[str, np.ndarray] = {}
        network['W1'] = n((2, 3), *(0.1, 0.3, 0.5, 0.2, 0.4, 0.6))
        network['b1'] = n((3,), *(0.1, 0.2, 0.3))
        network['W2'] = n((3, 2), *(0.1, 0.4, 0.2, 0.5, 0.3, 0.6))
        network['b2'] = n((2,), *(0.1, 0.2))
        network['W3'] = n((2, 2), *(0.1, 0.3, 0.2, 0.4))
        network['b3'] = n((2,), *(0.1, 0.2))

        return network
    
    def forward(network, x):
        W1, W2, W3 = network['W1'], network['W2'], network['W3']
        b1, b2, b3 = network['b1'], network['b2'], network['b3']

        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        z2 = sigmoid(a2)
        a3 = np.dot(z2, W3) + b3
        y = identity_function(a3)

        return y

