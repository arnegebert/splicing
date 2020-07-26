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
import csv

class ReconExon_EmbeddedDataLoader(BaseDataLoader):
    """
    PSI data loading demo using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        self.data_dir = data_dir
        start = time.time()
        print(f'starting loading of data')
        self.samples = []
        cons, low, high = [], [], []
        data_type = 'cass'
        if True:
            x_cons_data = np.load('data/dsc_reconstruction_exon/embedded_brain_cortex_cons.npy')
            hx_cas_data = np.load('data/dsc_reconstruction_exon/embedded_brain_cortex_high.npy')
            lx_cas_data = np.load('data/dsc_reconstruction_exon/embedded_brain_cortex_low.npy')

            # x_cons_data = x_cons_data[:10]
            # hx_cas_data = hx_cas_data[:10]
            # lx_cas_data = lx_cas_data[:10]

            # a = int(len(x_cons_data) / 10)
            # b = int(len(hx_cas_data) / 10)
            # c = int(len(lx_cas_data) / 10)

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
            total = a + (b + c) * d
            cons_perc = a / total
            print(f'Percentage of consecutive data: {cons_perc}')
            if cons_perc > 0.6 or cons_perc < 0.4:
                raise Exception('Unbalanced dataset')
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


            train = torch.tensor(train)
            # cons + low + high
            val_all = torch.tensor(test)
            # cons + low
            val_low = torch.tensor(lt)
            # cons + high
            val_high = torch.tensor(htest)

            # train = train
            # # cons + low + high
            # val_all = test
            # # cons + low
            # val_low = lt
            # # cons + high
            # val_high = htest
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
        print(f'Size training dataset: {len(train_dataset)}')
        print(f'Size mixed validation dataset: {len(val_all_dataset)}')
        print(f'Size low inclusion validation dataset: {len(val_low_dataset)}')
        print(f'Size high inclusion validation dataset: {len(val_high_dataset)}')
        self.dataset = (train_dataset, val_all_dataset, val_low_dataset, val_high_dataset)
        end = time.time()
        print('total time to load data: {} secs'.format(end - start))
        # samples = prepare_data()
        # self.dataset = TensorDataset(*samples)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers, dsc_cv=True)

def one_hot_decode(nt):
    if (nt == [1.0, 0, 0, 0]).all():
        return 'A'
    elif (nt == [0, 1.0, 0, 0]).all():
        return 'C'
    elif (nt == [0, 0, 1.0, 0]).all():
        return 'G'
    elif (nt == [0, 0, 0, 1.0]).all():
        return 'T'
    else: raise Exception('Unknown encoding. Decoding failed. ')

def decode_batch_of_seqs(batch):
    to_return = []
    for seq in batch:
        for encoding in seq:
            to_return.append(one_hot_decode(encoding))
    return to_return


# Constant values taken in reference from
# https://github.com/louadi/DSC/blob/master/training%20notebooks/cons_vs_es.ipynb
# Don't blame me ¯\_(ツ)_/¯
def extract_values_from_dsc_np_format(array):
    lifehack = 500000
    class_task = True
    if class_task:
        # classification
        labels = array[:lifehack, 140, 0]
    else:
        # psi value
        labels = array[:lifehack, -1, 4]

    start_seq, end_seq = array[:lifehack, :140, :4], array[:lifehack, 141:281, :4]
    start_seq, end_seq = decode_batch_of_seqs(start_seq), decode_batch_of_seqs(end_seq)
    lens = array[:lifehack, -1, 0:3]
    to_return = []
    # could feed my network data with 280 + 3 + 1 dimensions
    for start, end, len, label in zip(start_seq, end_seq, lens, labels):
        to_return.append(((start, end), T(len).float(), T(label).float()))
    return to_return

class DSCDataset(Dataset):
    """ Implementation of Dataset class for the synthetic dataset. """

    def __init__(self, samples):
        random.seed(0)
        # woooow, pretty proud of myself for figuring this out :>
        # random.shuffle(samples)
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

