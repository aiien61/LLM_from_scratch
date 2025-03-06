import torch
import numpy as np
from icecream import ic

print(torch.backends.mps.is_available())
print(torch.cuda.is_available())
ic(torch.__version__)

tensor_0d = torch.tensor(1)  # creates a 0 dimensional tensor by using integer
tensor_1d = torch.tensor([1, 2, 3])  # creates a 1 dimensional tensor by using an array
tensor_2d = torch.tensor([[1, 2], [3, 4]])  # creates a 2 dimensional tensor by using a matrix
tensor_3d = torch.tensor([[[1, 2], [3, 4]],
                          [[5, 6], [7, 8]]])  # create a 3 dimensional tensor by using a nested array

ic(tensor_0d)
ic(tensor_1d)
ic(tensor_2d)
ic(tensor_3d)

# 64 bit is default for integers
ic(tensor_1d.dtype)

# 32 bit is default for floating numbers
floatvec = torch.tensor([1.0, 2.0, 3.0])
ic(floatvec.dtype)

# Uses .to() to convert 64 bits to 32 bits
floatvec = tensor_1d.to(torch.float32)
ic(floatvec.dtype)

tensor_2d = torch.tensor([[1, 2, 3], [4, 5, 6]])
ic(tensor_2d)

ic(tensor_2d.shape)
ic(tensor_3d.shape)

# Reshaping
ic(tensor_2d.reshape(3, 2))  # reshapes 2d tensor with (2, 3) shape into a 2d tensor with (3, 2) shape 

# .view() is more often used in reshaping tensors
ic(tensor_2d.view(3, 2))

# Transposition
ic(tensor_2d.T)

# Matrix multiplication
ic(tensor_2d.matmul(tensor_2d.T))
ic(tensor_2d @ tensor_2d.T)