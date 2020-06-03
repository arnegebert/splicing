from torchvision import datasets, transforms
from base import BaseDataLoader
from torch.utils.data import Dataset
import ast
import time
from torch import as_tensor as T
import pickle


class NaivePSIDataLoader(BaseDataLoader):
    """
    PSI data loading demo using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        trsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.data_dir = data_dir
        self.dataset = NaivePSIDataset(path=data_dir)
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

class NaivePSIDataset(Dataset):
    """ Implementation of Dataset class for the synthetic dataset. """

    def __init__(self, path, transform=None):
        self.path = path
        start = time.time()
        print(f'starting loading of data')
        self.samples = []

        with open(self.path, 'r') as f:
            for i, l in enumerate(f):
                j, start_seq, end_seq, psi = l.split(',')
                psi = float(psi[:-1])
                sample = (T((encode_seq(start_seq), encode_seq(end_seq))), T(psi))
                self.samples.append(sample)
        end = time.time()
        print('total time to load data: {} secs'.format(end - start))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]