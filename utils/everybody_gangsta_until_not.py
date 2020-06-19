import torch
import random

random.seed(0)

x = [1, 2, 3]
xt = torch.tensor(x)

random.shuffle(xt)
random.shuffle(x)

print(x)
print(xt)