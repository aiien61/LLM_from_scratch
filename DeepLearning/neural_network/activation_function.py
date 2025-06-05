import numpy as np
import matplotlib.pylab as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

def identity_function(x):
    return x

def softmax(a: np.ndarray):
    if not isinstance(a, np.ndarray):
        a = np.array(a)

    c = np.max(a)
    exp_a: np.ndarray = np.exp(a - c)
    return exp_a / np.sum(exp_a)

if __name__ == "__main__":
    a = [1010, 1000, 990]
    y = softmax(a)
    print(y)
    print(np.sum(y))

