import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel
import copy
import torch
import numpy as np
import math
from torch.autograd import Variable
from torch import as_tensor as T, Tensor
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from sklearn import metrics

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Training with device={device}')

lr = 5e-4
batch =256
epochs = 150

class MLP(torch.nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(3, 4)
        self.fc2 = nn.Linear(4, 1)

    def forward(self, lens):
        x = F.relu(self.fc1(lens))
        x = torch.sigmoid(self.fc2(x))
        return x

model = MLP()
optimizer = Adam(model.parameters(), lr=lr)

x_cons_data = np.load('data/hexevent/x_cons_data.npy')
hx_cas_data = np.load('data/hexevent/x_cas_data_high.npy')
lx_cas_data = np.load('data/hexevent/x_cas_data_low.npy')
x_cons_data[:, -1, 4] = 1

a = int(x_cons_data.shape[0] / 10)
b = int(hx_cas_data.shape[0] / 10)
c = int(lx_cas_data.shape[0] / 10)

s = 0
# 9 folds for training
train = x_cons_data[:a * s]
train = np.concatenate((train, x_cons_data[a * (s + 1):]), axis=0)

d = int((9 * a) / (9 * (b + c)))
d = max(1, d)
print(d)
classification_task = False
for i in range(d):  # range(1)
    train = np.concatenate((train, hx_cas_data[:b * s]), axis=0)
    train = np.concatenate((train, hx_cas_data[b * (s + 1):]), axis=0)

    train = np.concatenate((train, lx_cas_data[:c * s]), axis=0)
    train = np.concatenate((train, lx_cas_data[c * (s + 1):]), axis=0)

np.random.seed(0)
np.random.shuffle(train)

# 1 fold for testing

htest = np.concatenate((hx_cas_data[b * s:b * (s + 1)], x_cons_data[a * s:a * (s + 1)]), axis=0)
lt = np.concatenate((lx_cas_data[c * s:c * (s + 1)], x_cons_data[a * s:a * (s + 1)]), axis=0)

test = htest
test = np.concatenate((test, lx_cas_data[c * s:c * (s + 1)]), axis=0)

cons_test = x_cons_data[a * s:a * (s + 1)]
cas_test = np.concatenate((lx_cas_data[c * s:c * (s + 1)], hx_cas_data[b * s:b * (s + 1)]))

def extract_values_from_dsc_np_format(array):
    lifehack = 500000
    class_task = True
    if class_task:
        # classification
        label = array[:lifehack, 140, 0]
    else:
        # psi value
        label = array[:lifehack, -1, 4]
    start_seq, end_seq = array[:lifehack, :140, :4], array[:lifehack, 141:281, :4]
    lens = array[:lifehack, -1, 0:3]
    to_return = []
    # could feed my network data with 280 + 3 + 1 dimensions
    for s, e, l, p in zip(start_seq, end_seq, lens, label):
        to_return.append((T((s, e)).float(), T(l).float(), T(p).float()))
    return to_return

class DSCDataset(Dataset):
    """ Implementation of Dataset class for the synthetic dataset. """

    def __init__(self, samples):
        # random.seed(0)
        # random.shuffle(samples)
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

def auc(output, target):
    with torch.no_grad():
        return metrics.roc_auc_score(target.cpu(), output.cpu())

train = extract_values_from_dsc_np_format(train)
# cons + low + high
val_all = extract_values_from_dsc_np_format(test)
# cons + low
val_low = extract_values_from_dsc_np_format(lt)
# cons + high
val_high = extract_values_from_dsc_np_format(htest)
train_dataset = DSCDataset(train)
val_all_dataset = DSCDataset(val_all)
val_low_dataset = DSCDataset(val_low)
val_high_dataset = DSCDataset(val_high)

print('Loaded data')
init_kwargs = {
            'batch_size': batch,
            'shuffle': True,
            'num_workers': 0,
            'drop_last': False
        }

train_dataloader = DataLoader(dataset=train_dataset, **init_kwargs)
val_dataloader = DataLoader(dataset=val_all_dataset, **init_kwargs)

for epoch in range(epochs):
    for batch_idx, data in enumerate(train_dataloader):
        lens, target = data[:, 280, :3], data[:, 280, 3]
        lens, target = lens.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(lens)
        loss = F.binary_cross_entropy(output.view(-1), target)
        loss.backward()
        optimizer.step()

        auc = auc(output, target)

        if batch_idx==0:
            print(f'Loss: {loss}')
            print(f'AUC: {auc}')



