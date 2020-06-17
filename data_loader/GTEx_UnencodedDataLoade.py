from torchvision import datasets, transforms
from base import BaseDataLoader
from torch.utils.data import Dataset
import ast
import time
from torch import as_tensor as T
import pickle


class GTEx_UnencodedDataLoader(BaseDataLoader):
    """
    PSI data loading demo using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        self.data_dir = data_dir
        self.dataset = GTEx_UnencodedDataset(path=data_dir)
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

class GTEx_UnencodedDataset(Dataset):
    """ Implementation of Dataset class for the synthetic dataset. """

    def __init__(self, path, transform=None):
        self.path = path
        start = time.time()
        print(f'starting loading of data')
        self.samples = []

        avg_len = 7853.118899261425
        std_len = 23917.691461462917
        constitutive_level = 0.90

        low = 0
        medium = 0
        high = 0
        cons = 0

        with open(self.path, 'r') as f:
            for i, l in enumerate(f):
                j, start_seq, end_seq, psi = l.split(',')
                s, e = j.split('_')[1:3]
                s, e = int(s), int(e)
                psi = float(psi[:-1])
                is_constitutive = psi >= constitutive_level
                is_constitutive = T(float(is_constitutive))
                if psi <= 0.2: low += 1
                if 0.2 < psi <= 0.75: medium += 1
                if psi > 0.75 and psi < constitutive_level: high += 1
                if psi > constitutive_level: cons += 1
                l1 = (e-s-avg_len)/std_len
                l1 = T(l1).float()
                sample = ((start_seq, end_seq), l1, is_constitutive)
                self.samples.append(sample)
        end = time.time()
        print('total time to load data: {} secs'.format(end - start))
        print(f'low: {low}')
        print(f'medium: {medium}')
        print(f'high: {high}')
        print(f'cons: {cons}')
        print(f'all: {low+high+cons}')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]