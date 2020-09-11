import torch
import random
import numpy as np

"""
Code to reproduce a PyTorch bug which I found during the thesis 
see: https://github.com/pytorch/pytorch/issues/18934
"""

random.seed(0)

x = [1, 2, 3]
xt = torch.tensor(x)

np.random.shuffle(xt)
np.random.shuffle(x)

print(xt)
print(x)
