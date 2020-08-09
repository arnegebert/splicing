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

class Comparison_DataLoader(BaseDataLoader):
    """
    Implements all three use cases for HIPSCI NotNeuron:
    1. Train on lib1, test on lib1
    2. Train on lib1, test on lib2
    3. Train on lib1, test on lib from other individual
    4. Train on lib1, test on lib from other individual other tissue
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True,
                 classification=True, classification_treshold=0.95, cross_validation_split=0):
        self.data_dir = data_dir
        self.classification = classification
        self.class_threshold = classification_treshold
        self.cross_validation_split = cross_validation_split
        start = time.time()
        print(f'starting loading of data')
        self.samples = []

        if data_dir:
            raise Exception('reeeeeeeeeeeeeeeeeeeee')
        else:
            x_cons_data = np.load('data/not_neuron/exon/cons.npy')
            hx_cas_data = np.load('data/not_neuron/exon/high_bezi1.npy')
            lx_cas_data = np.load('data/not_neuron/exon/low_bezi1.npy')

            diff_lib_x_cons_data = np.load('data/not_neuron/exon/cons.npy')
            diff_lib_hx_cas_data = np.load('data/not_neuron/exon/high_bezi2.npy')
            diff_lib_lx_cas_data = np.load('data/not_neuron/exon/low_bezi2.npy')

            diff_indv_x_cons_data = np.load('data/not_neuron/exon/cons.npy')
            diff_indv_hx_cas_data = np.load('data/not_neuron/exon/high_lexy2.npy')
            diff_indv_lx_cas_data = np.load('data/not_neuron/exon/low_lexy2.npy')

            diff_tissue_x_cons_data = np.load('data/hipsci_majiq/exon/cons.npy')
            diff_tissue_hx_cas_data = np.load('data/hipsci_majiq/exon/high.npy')
            diff_tissue_lx_cas_data = np.load('data/hipsci_majiq/exon/low.npy')





        self.dataset_same = self.cross_validation(x_cons_data, lx_cas_data, hx_cas_data)
        #train_dataset, val_all_dataset, val_low_dataset, val_high_dataset = self.cross_validation(x_cons_data, lx_cas_data, hx_cas_data)
        dataset_diff_lib = self.cross_validation(diff_lib_x_cons_data, diff_lib_lx_cas_data, diff_lib_hx_cas_data)
        dataset_diff_indv = self.cross_validation(diff_indv_x_cons_data, diff_indv_lx_cas_data, diff_indv_hx_cas_data)
        dataset_diff_tissue = self.cross_validation(diff_tissue_x_cons_data, diff_tissue_lx_cas_data, diff_tissue_hx_cas_data)
        val_datasets_diff_lib = dataset_diff_lib[1:]
        val_datasets_diff_indv = dataset_diff_indv[1:]
        val_datasets_diff_tissue = dataset_diff_tissue[1:]

        # flatten dataset elements
        extra_val_datasets = [sample for dataset in
                              [val_datasets_diff_lib, val_datasets_diff_indv, val_datasets_diff_tissue]
                              for sample in dataset]

        end = time.time()
        print('total time to load data: {} secs'.format(end - start))
        super().__init__(self.dataset_same, batch_size, shuffle, validation_split, num_workers, dsc_cv=True,
                         extra_val_datasets=extra_val_datasets)

    def cross_validation(self, x_cons_data, lx_cas_data, hx_cas_data):
        if self.classification:
            x_cons_data[:, 280, 3] = (x_cons_data[:, 280, 3] >= self.class_threshold).astype(np.float32)
            hx_cas_data[:, 280, 3] = (hx_cas_data[:, 280, 3] >= self.class_threshold).astype(np.float32)
            lx_cas_data[:, 280, 3] = (lx_cas_data[:, 280, 3] >= self.class_threshold).astype(np.float32)

        # cons = extract_values_from_dsc_np_format(x_cons_data)
        # low = extract_values_from_dsc_np_format(hx_cas_data)
        # high = extract_values_from_dsc_np_format(lx_cas_data)
        a = int(x_cons_data.shape[0] / 10)
        b = int(hx_cas_data.shape[0] / 10)
        c = int(lx_cas_data.shape[0] / 10)

        s = self.cross_validation_split
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

        train_dataset = NN_Dataset(train)
        val_all_dataset = NN_Dataset(val_all)
        val_low_dataset = NN_Dataset(val_low)
        val_high_dataset = NN_Dataset(val_high)
        return train_dataset, val_all_dataset, val_low_dataset, val_high_dataset

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



class NN_Dataset(Dataset):
    """ Implementation of Dataset class for the synthetic dataset. """

    def __init__(self, samples):
        self.samples = torch.tensor(samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]
