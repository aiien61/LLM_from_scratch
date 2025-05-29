import numpy as np
from icecream import ic
from rich import print

x: np.ndarray = np.array([1, 2, 3])
ic(x)
ic(x.__class__)
ic(isinstance(x, np.ndarray))
ic(x.shape)
ic(x.ndim)

print("-"*30)

W: np.ndarray = np.array([[1,2,3], [4,5,6]])
ic(W)
ic(W.__class__)
ic(W.shape)
ic(W.ndim)

print("-"*30)

Q: np.ndarray = np.array(
    [
        [[1, 2, 3], [4, 5, 6]],
        [[7, 8, 9], [10, 11, 12]]
    ]
)
ic(Q)
ic(Q.__class__)  # numpy.ndarray
ic(Q.shape)  # (2, 2, 3)
ic(Q.ndim)  # 3

print("-"*30)

# element-wise operations
W: np.ndarray = np.array([[1, 2, 3], [4, 5, 6]])
X: np.ndarray = np.array([[0, 1, 2], [3, 4, 5]])

ic(W)
ic(X)

ic(W + X)
ic(W * X)

# broadcasting
A: np.ndarray = np.array([[1, 2], [3, 4]])

ic(A)
ic(A * 10)

print("-"*30)

A: np.ndarray = np.array([[1, 2], [3, 4]])
b: np.ndarray = np.array([10, 20])

ic(A)
ic(A.shape)
ic(A.ndim)
ic(b)
ic(b.shape)
ic(b.ndim)
ic(A * b)

print("-"*30)

# dot-product
a: np.ndarray = np.array([1, 2, 3])
b: np.ndarray = np.array([4, 5, 6])

ic(a)
ic(a.shape)
ic(a.ndim)
ic(b)
ic(b.shape)
ic(b.ndim)
ic(np.dot(a, b))

print("-"*30)

# matrix multiplication
A: np.ndarray = np.array([[1, 2], [3, 4]])
B: np.ndarray = np.array([[5, 6], [7, 8]])

ic(A)
ic(A.shape)
ic(A.ndim)
ic(B)
ic(B.shape)
ic(B.ndim)
ic(np.dot(A, B))
