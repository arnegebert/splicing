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

class HEXEvent_DataLoader(BaseDataLoader):
    """
    PSI data loading demo using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True,
                 cross_validation_split=0):
        self.data_dir = data_dir
        start = time.time()
        print(f'starting loading of data')
        self.samples = []
        cons, low, high = [], [], []
        data_type = 'cass'
        if True:
            x_cons_data = np.load('data/hexevent/cons.npy')
            hx_cas_data = np.load('data/hexevent/high.npy')
            lx_cas_data = np.load('data/hexevent/low.npy')
            # cons = extract_values_from_dsc_np_format(x_cons_data)
            # low = extract_values_from_dsc_np_format(hx_cas_data)
            # high = extract_values_from_dsc_np_format(lx_cas_data)
            x_cons_data[:,-1,4] = 1
            a = int(x_cons_data.shape[0] / 10)
            b = int(hx_cas_data.shape[0] / 10)
            c = int(lx_cas_data.shape[0] / 10)

            s = cross_validation_split
            # 9 folds for training
            train = x_cons_data[:a * s]
            train = np.concatenate((train, x_cons_data[a * (s + 1):]), axis=0)

            d = int((9 * a) / (9 * (b + c)))
            d = max(1, d)
            total = a + (b + c) * d
            cons_perc = a / total
            print(f'Percentage of consecutive data: {cons_perc}')
            if cons_perc > 0.6 or cons_perc < 0.4:
                raise Exception('Unbalanced dataset')
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


            train = extract_values_from_dsc_np_format(train)
            # cons + low + high
            val_all = extract_values_from_dsc_np_format(test)
            # cons + low
            val_low = extract_values_from_dsc_np_format(lt)
            # cons + high
            val_high = extract_values_from_dsc_np_format(htest)

            # return train, test, htest, lt, cons_test, cas_test

        else:
            with open('data/hexevent/all_cons_filtered_class.csv', 'r') as f:
                for i, l in enumerate(f):
                    j, start_seq, end_seq, psi, l1, l2, l3 = l.split('\t')
                    psi, l1, l2, l3 = float(psi), float(l1), float(l2), float(l3[:-1])
                    seqs = T((encode_seq(start_seq), encode_seq(end_seq)))
                    lens = T((l1, l2, l3))
                    psi = T(psi)
                    sample = (seqs, lens, psi)
                    cons.append(sample)

            with open(f'data/hexevent/low_{data_type}_filtered_class.csv', 'r') as f:
                for i, l in enumerate(f):
                    j, start_seq, end_seq, psi, l1, l2, l3 = l.split('\t')
                    psi, l1, l2, l3 = float(psi), float(l1), float(l2), float(l3[:-1])
                    seqs = T((encode_seq(start_seq), encode_seq(end_seq)))
                    lens = T((l1, l2, l3))
                    psi = T(psi)
                    sample = (seqs, lens, psi)
                    low.append(sample)

            with open(f'data/hexevent/high_{data_type}_filtered_class.csv', 'r') as f:
                for i, l in enumerate(f):
                    j, start_seq, end_seq, psi, l1, l2, l3 = l.split('\t')
                    psi, l1, l2, l3 = float(psi), float(l1), float(l2), float(l3[:-1])
                    seqs = T((encode_seq(start_seq), encode_seq(end_seq)))
                    lens = T((l1, l2, l3))
                    psi = T(psi)
                    sample = (seqs, lens, psi)
                    high.append(sample)

            ratio = int(len(cons) / (len(low) + len(high))) + 1

            len_cons, len_low, len_high = int(len(cons)*validation_split), int(len(low)*validation_split),\
                                          int(len(high)*validation_split)
            cons_val, cons_train = cons[:len_cons], cons[len_cons:]
            lval, ltrain = low[:len_low], low[len_low:]
            hval, htrain = high[:len_high], high[len_high:]

            train = cons_train
            for _ in range(ratio):
                train.extend(ltrain)
                train.extend(htrain)

            val_all = cons_val + lval + hval
            val_low = cons_val + lval
            val_high = cons_val + hval

        # random.seed(0)
        # random.shuffle(train)
        # random.shuffle(val_all)
        # random.shuffle(val_low)
        # random.shuffle(val_high)

        train_dataset = DSCDataset(train)
        val_all_dataset = DSCDataset(val_all)
        val_low_dataset = DSCDataset(val_low)
        val_high_dataset = DSCDataset(val_high)
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

# Constant values taken in reference from
# https://github.com/louadi/DSC/blob/master/training%20notebooks/cons_vs_es.ipynb
# Don't blame me ¯\_(ツ)_/¯
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