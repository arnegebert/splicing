from torchvision import datasets, transforms
from base import BaseDataLoader
from torch.utils.data import Dataset
import ast
import time
from torch import as_tensor as T
import pickle
import numpy as np
import torch

class GTEx_EmbeddedDataLoader(BaseDataLoader):
    """
    PSI data loading demo using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True,
                 cross_validation_split=0):
        # todo: cross_validation_split parameter ignored
        self.data_dir = data_dir
        self.dataset = GTEx_EmbeddedDataset(path=data_dir)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

def one_hot_encode(nt):
    if nt == 'A' or nt == 'a':
        return [1.0, 0, 0, 0]
    elif nt == 'C' or nt == 'c':
        return [0, 1.0, 0, 0]
    elif nt == 'G' or nt == 'g':
        return [0, 0, 1.0, 0]
    elif nt == 'T' or nt == 't':
        return [0, 0, 0, 1.0]

def encode_seq(seq):
    encoding = []
    for nt in seq:
        encoding.append(one_hot_encode(nt))
    return encoding

class GTEx_EmbeddedDataset(Dataset):
    """ Implementation of Dataset class for the synthetic dataset. """

    def __init__(self, path, transform=None):
        startt = time.time()
        print(f'starting loading of data')
        self.samples = np.load(path)
        np.random.seed(0)
        np.random.shuffle(self.samples)

        self.samples = torch.from_numpy(self.samples)


        endt = time.time()
        print(f'Took {endt-startt} to load data')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]