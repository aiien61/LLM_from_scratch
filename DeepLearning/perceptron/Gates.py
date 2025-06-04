import numpy as np

# single-layered perceptron
def AND(x1, x2):
    x: np.ndarray = np.array([x1, x2])
    w: np.ndarray = np.array([0.5, 0.5])
    b: float = -0.8
    tmp: float = np.sum(x * w) + b
    return 0 if tmp <= 0 else 1

# single-layered perceptron
def NAND(x1, x2):
   x: np.ndarray = np.array([x1, x2])
   w: np.ndarray = np.array([-0.5, -0.5])
   b: float = 0.8
   tmp: float = np.sum(x * w) + b
   return 0 if tmp <= 0 else 1

# single-layered perceptron
def OR(x1, x2):
    x: np.ndarray = np.array([x1, x2])
    w: np.ndarray = np.array([0.5, 0.5])
    b: float = -0.3
    tmp: float = np.sum(x * w) + b
    return 0 if tmp <= 0 else 1

# multi-layered perceptron
def XOR(x1, x2):
    s1: int = NAND(x1, x2)
    s2: int = OR(x1, x2)
    return AND(s1, s2)
