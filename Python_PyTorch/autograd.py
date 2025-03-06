"""Takes simple logistic regression classifier as an example to demonstrates how to use PyTorch autograd
"""
import torch
import torch.nn.functional as F
from torch.autograd import grad
from icecream import ic

y = torch.tensor([1.0])  # target label  i.e. The Answer
x1 = torch.tensor([1.1])  # features
w1 = torch.tensor([2.2], requires_grad=True)  # weights
b = torch.tensor([0.0], requires_grad=True)   # bias unit

z = x1 * w1 + b
a = torch.sigmoid(z)  # Uses sigmoid as the activation function

loss = F.binary_cross_entropy(a, y)
ic(loss)


grad_L_w1 = grad(loss, w1, retain_graph=True)
grad_L_b = grad(loss, b, retain_graph=True)

ic(grad_L_w1)
ic(grad_L_b)

loss.backward()
ic(w1.grad)
ic(b.grad)
