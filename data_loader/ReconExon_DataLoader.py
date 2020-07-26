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

eps = 1e-3
def kms(lens_gai, lens_fake):
    l1, l2, l3 = lens_gai
    l1, l2, l3 = float(l1), float(l2), float(l3)
    l4, l5, l6 = lens_fake
    l4, l5, l6 = float(l4), float(l5), float(l6)
    b1 = abs(l1 - l4 ) < eps
    b2 = abs(l2 - l5 ) < eps
    b3 = abs(l3 - l6 ) < eps
    return b1 and b2 and b3
    #return (l1, l2, l3) == (l4, l5, l6)

class ReconExon_DataLoader(BaseDataLoader):
    """
    PSI data loading demo using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True,
                 classification=True, classification_treshold=0.95):
        self.data_dir = data_dir
        start = time.time()
        print(f'starting loading of data')
        self.samples = []
        cons, low, high = [], [], []
        data_type = 'cass'
        if True:

            x_cons_data = np.load('data/dsc_reconstruction_exon/brain_cortex_cons.npy')
            hx_cas_data = np.load('data/dsc_reconstruction_exon/brain_cortex_high.npy')
            lx_cas_data = np.load('data/dsc_reconstruction_exon/brain_cortex_low.npy')

            # lens, target = hx_cas_data[:, 280, :3], hx_cas_data[:, 280, 3]
            # xxx = torch.tensor([ 0.60716164,  0.80365247, -0.22670029])
            # xxx2 = np.array([ 0.60716164,  0.80365247, -0.22670029])
            # for i, ls in enumerate(lens):
            #     # if kms(xxx2, ls):
            #     if np.sum(np.isclose(xxx2,ls)) == 3:
            #         # if torch.sum(xxx == ls) == torch.tensor(3):
            #         print(i)
            #         print('subdued')

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

        train_dataset = ReconExon_Dataset(train)
        val_all_dataset = ReconExon_Dataset(val_all)
        val_low_dataset = ReconExon_Dataset(val_low)
        val_high_dataset = ReconExon_Dataset(val_high)
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



class ReconExon_Dataset(Dataset):
    """ Implementation of Dataset class for the synthetic dataset. """

    def __init__(self, samples):
        self.samples = torch.tensor(samples)

        # lens, target = samples[:, 280, :3], samples[:, 280, 3]
        # xxx = torch.tensor([-0.23017790490661963, -0.16999491874342373, -0.1983459465537275])
        # xxx2 = np.array([-0.23017790490661963, -0.16999491874342373, -0.1983459465537275])
        # for i, ls in enumerate(lens):
        #     if np.sum(xxx2 == ls) == 3:
        #     # if torch.sum(xxx == ls) == torch.tensor(3):
        #         print(i)
        #         print('subdued happyness')


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]
