from torchvision import datasets, transforms
from base import BaseDataLoader
from torch.utils.data import Dataset, TensorDataset
import ast
import time
from torch import as_tensor as T, Tensor
import pickle
import torch
import random
import numpy as np

class HIPSCI_SUPPA_DataLoader(BaseDataLoader):
    """
    PSI data loading demo using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True,
                 classification=True, classification_treshold=0.95):
        self.data_dir = data_dir
        start = time.time()
        print(f'starting loading of data')
        self.samples = []
        if data_dir:
            x_cons_data = np.load(f'{data_dir}/cons.npy')
            hx_cas_data = np.load(f'{data_dir}/high.npy')
            lx_cas_data = np.load(f'{data_dir}/low.npy')
        else:
            x_cons_data = np.load('data/hipsci_suppa/cons.npy')
            hx_cas_data = np.load('data/hipsci_suppa/high.npy')
            lx_cas_data = np.load('data/hipsci_suppa/low.npy')

        if classification:
            x_cons_data[:, 280, 3] = (x_cons_data[:, 280, 3] >= classification_treshold).astype(np.float32)
            hx_cas_data[:, 280, 3] = (hx_cas_data[:, 280, 3] >= classification_treshold).astype(np.float32)
            lx_cas_data[:, 280, 3] = (lx_cas_data[:, 280, 3] >= classification_treshold).astype(np.float32)

        # cons = extract_values_from_dsc_np_format(x_cons_data)
        # low = extract_values_from_dsc_np_format(hx_cas_data)
        # high = extract_values_from_dsc_np_format(lx_cas_data)
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
        for i in range(d): #range(1)
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


        train = train
        # cons + low + high
        val_all = test
        # cons + low
        val_low = lt
        # cons + high
        val_high = htest

        print(f'Size training dataset: {len(train)}')
        print(f'Size mixed validation dataset: {len(val_all)}')
        print(f'Size low inclusion validation dataset: {len(val_low)}')
        print(f'Size high inclusion validation dataset: {len(val_high)}')

        # return train, test, htest, lt, cons_test, cas_test

        # random.seed(0)
        # random.shuffle(train)
        # random.shuffle(val_all)
        # random.shuffle(val_low)
        # random.shuffle(val_high)

        train_dataset = HIPSCI_SUPPA_Dataset(train)
        val_all_dataset = HIPSCI_SUPPA_Dataset(val_all)
        val_low_dataset = HIPSCI_SUPPA_Dataset(val_low)
        val_high_dataset = HIPSCI_SUPPA_Dataset(val_high)
        self.dataset = (train_dataset, val_all_dataset, val_low_dataset, val_high_dataset)
        end = time.time()
        print('total time to load data: {} secs'.format(end - start))
        # samples = prepare_data()
        # self.dataset = TensorDataset(*samples)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers, dsc_cv=True)

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


class HIPSCI_SUPPA_Dataset(Dataset):
    """ Implementation of Dataset class for the synthetic dataset. """

    def __init__(self, samples):
        self.samples = torch.tensor(samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]
