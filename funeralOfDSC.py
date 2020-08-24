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
seed = 3

torch.manual_seed(seed)
np.random.seed(seed)

# Epoch 148:
# Training loss: 0.4518568434706136
# Training AUC: 0.8945468445022248
# Validation loss: 0.5131238667588485
# Validation AUC: 0.9000702936870301
class MLP100(BaseModel):
    def __init__(self):
        super(MLP100, self).__init__()
        self.fc1 = nn.Linear(3, 20, bias=True)
        self.fc2 = nn.Linear(20, 1, bias=False)

    def forward(self, lens):
        x = F.relu(self.fc1(lens))
        x = torch.sigmoid(self.fc2(x))
        return x

# Epoch 146:
# Training loss: 0.49831184333768386
# Training AUC: 0.859443620291274
# Validation loss: 0.5369397150842767
# Validation AUC: 0.8544548156084086
class MLP20(BaseModel):
    def __init__(self):
        super(MLP20, self).__init__()
        self.fc1 = nn.Linear(3, 4, bias=True)
        self.fc2 = nn.Linear(4, 1, bias=False)

    def forward(self, lens):
        x = F.relu(self.fc1(lens))
        x = torch.sigmoid(self.fc2(x))
        return x

class MLPLinear(BaseModel):
    def __init__(self):
        super(MLPLinear, self).__init__()
        self.fc1 = nn.Linear(3, 4, bias=True)
        self.fc2 = nn.Linear(4, 1, bias=False)

    def forward(self, lens):
        x = self.fc1(lens)
        x = torch.sigmoid(self.fc2(x))
        return x

model = MLP100().to(device)
print(model.__str__())
# optimizer = Adam(model.parameters(), lr=lr, weight_decay=0, amsgrad=True)
optimizer = RMSprop(model.parameters(), lr=lr)

# original data from DSC
x_cons_data = np.load('data/hexevent/x_cons_data.npy').astype(np.float32)
hx_cas_data = np.load('data/hexevent/x_cas_data_high.npy').astype(np.float32)
lx_cas_data = np.load('data/hexevent/x_cas_data_low.npy').astype(np.float32)
# setting 'psi' of constitutive data to 1
x_cons_data[:, -1, 4] = 1

# pre-processing / shuffling methods taken from DSC GitHub
a = int(x_cons_data.shape[0] / 10)
b = int(hx_cas_data.shape[0] / 10)
c = int(lx_cas_data.shape[0] / 10)

s = 0
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

def get_lens_target_from_dsc_format(data):
    return data[:, -1, :3], data[:, 140, 0]

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
        lens, target = get_lens_target_from_dsc_format(data)
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
            lens, target = get_lens_target_from_dsc_format(data)
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
