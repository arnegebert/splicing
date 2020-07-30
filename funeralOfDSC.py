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
from torch.optim import Adam, RMSprop
from sklearn import metrics
import time

start = time.time()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Training with device={device}')

lr = 5e-3
batch =256
epochs = 150
seed = 0

torch.manual_seed(0)
np.random.seed(0)

class MLP(torch.nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(3, 12, bias=True)
        self.fc2 = nn.Linear(12, 1, bias=False)

    def forward(self, lens):
        x = F.relu(self.fc1(lens))
        x = torch.sigmoid(self.fc2(x))
        return x

    def __str__(self):
        """
        Model prints with number of trainable parameters
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)

model = MLP().to(device)
print(model.__str__())
optimizer = Adam(model.parameters(), lr=lr, weight_decay=0, amsgrad=True)
optimizer = RMSprop(model.parameters(), lr=lr)#, lr=lr, weight_decay=0, amsgrad=True)

x_cons_data = np.load('data/hexevent/x_cons_data.npy').astype(np.float32)
hx_cas_data = np.load('data/hexevent/x_cas_data_high.npy').astype(np.float32)
lx_cas_data = np.load('data/hexevent/x_cas_data_low.npy').astype(np.float32)
x_cons_data[:, -1, 4] = 1

a = int(x_cons_data.shape[0] / 10)
b = int(hx_cas_data.shape[0] / 10)
c = int(lx_cas_data.shape[0] / 10)

s = cross_validation_split
# 9 folds for training
train = x_cons_data[:a * s]
train = np.concatenate((train, x_cons_data[a * (s + 1):]), axis=0)

d = int((9 * a) / (9 * (b + c)))
d = max(1, d)
# d = 1
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
    lens = array[:lifehack, -1, :3]
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

# train = extract_values_from_dsc_np_format(train)
# # cons + low + high
# val_all = extract_values_from_dsc_np_format(test)
# # cons + low
# val_low = extract_values_from_dsc_np_format(lt)
# # cons + high
# val_high = extract_values_from_dsc_np_format(htest)

train = train
# cons + low + high
val_all = test
# cons + low
val_low = lt
# cons + high
val_high = htest

train_dataset = DSCDataset(train)
val_all_dataset = DSCDataset(val_all)
val_low_dataset = DSCDataset(val_low)
val_high_dataset = DSCDataset(val_high)

print('Loaded data')
init_kwargs = {
            'batch_size': batch,
            'shuffle': True,
            'num_workers': 0,
            'drop_last': True
        }

train_dataloader = DataLoader(dataset=train_dataset, **init_kwargs)
val_dataloader = DataLoader(dataset=val_all_dataset, **init_kwargs)

for epoch in range(epochs):
    batch_loss, batch_auc = 0, 0
    print('-'*40)
    print(f'Epoch {epoch}:')
    for batch_idx, data in enumerate(train_dataloader):
        lens, target = data[:, -1, :3], data[:, 140, 0]
        lens, target = lens.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(lens)
        loss = F.binary_cross_entropy(output.view(-1), target)
        loss.backward()
        optimizer.step()

        batch_loss += loss.item()
        batch_auc += auc(output, target)
    batch_loss /= len(train_dataloader)
    batch_auc /= len(train_dataloader)

    print(f'Training loss: {batch_loss}')
    print(f'Training AUC: {batch_auc}')

    with torch.no_grad():
        batch_loss, batch_auc = 0, 0
        for batch_idx, data in enumerate(val_dataloader):
            lens, target = data[:, -1, :3], data[:, 140, 0]
            lens, target = lens.to(device), target.to(device)
            output = model(lens)
            loss = F.binary_cross_entropy(output.view(-1), target)

            batch_loss += loss.item()
            batch_auc += auc(output, target)

        batch_loss /= len(val_dataloader)
        batch_auc /= len(val_dataloader)

        print(f'Validation loss: {batch_loss}')
        print(f'Validation AUC: {batch_auc}')


end = time.time()
print(f'Running time: {end-start} s')
