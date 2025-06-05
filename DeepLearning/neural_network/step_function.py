import numpy as np
import matplotlib.pylab as plt
from rich import print

def step_function(x):
    return np.array(x > 0, dtype=np.int64)

if __name__ == "__main__":
    x: np.ndarray = np.arange(-5, 5, 0.1)
    y: np.ndarray = step_function(x)
    plt.plot(x, y)
    plt.ylim(-0.1, 1.1)
    plt.show()