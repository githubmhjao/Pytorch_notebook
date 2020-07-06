# Large amount of credit goes to:
# https://jovian.ml/aakashns/01-pytorch-basics

import torch


# ============================================================= #
#                            Tensors                            #
# ============================================================= #

t1 = torch.tensor(4.)
t1.dtype                 # torch.float32
t1.shape                 # torch.Size([])

t2 = torch.tensor(4)
t2.dtype                 # torch.int64

t3 = torch.tensor([1., 2, 3, 4])
t3.shape                 # torch.Size([4])

t4 = torch.tensor([[5., 6],
                   [7,  8],
                   [9, 10]])
t4.shape                 # torch.Size([3, 2])


# ============================================================= #
#                Tensor Operations and Gradients                #
# ============================================================= #

x = torch.tensor(3.)
w = torch.tensor(4., requires_grad=True)
b = torch.tensor(5., requires_grad=True)

y = w * x + b

# Compute derivatives of y related to other variables
y.backward()

# Display gradients
x.grad                 # None
w.grad                 # tensor(3.)
b.grad                 # tensor(1.)
