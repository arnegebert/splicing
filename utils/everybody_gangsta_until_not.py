import torch
import random
import numpy as np

random.seed(0)

x = [1, 2, 3]
xt = torch.tensor(x)

# random.shuffle(xt)
# random.shuffle(x)
# print(x)
# print(xt)

np.random.shuffle(xt)
np.random.shuffle(x)

print(xt)
print(x)
