import numpy as np
from typing import Tuple

def n(shape: Tuple[int], *data):
    return np.array(data).reshape(shape)
