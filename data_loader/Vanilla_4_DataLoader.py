import time

import numpy as np
import torch
from torch.utils.data import Dataset

from base import BaseDataLoader


class Vanilla_4_DataLoader(BaseDataLoader):
    """
    PSI data loading demo using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True,
                 classification=True, classification_threshold=0.95, cross_validation_split=0):
        self.data_dir = data_dir
        start = time.time()
        print(f'starting loading of data')
        self.samples = []

        if data_dir:
            x_cons_data = np.load(f'{data_dir}/cons_4.npy')
            hx_cas_data = np.load(f'{data_dir}/high_4.npy')
            lx_cas_data = np.load(f'{data_dir}/low_4.npy')
        else:
            raise Exception('No data directory given')

        if classification:
            x_cons_data[:, 560, 3] = (x_cons_data[:, 560, 3] >= classification_threshold).astype(np.float32)
            hx_cas_data[:, 560, 3] = (hx_cas_data[:, 560, 3] >= classification_threshold).astype(np.float32)
            lx_cas_data[:, 560, 3] = (lx_cas_data[:, 560, 3] >= classification_threshold).astype(np.float32)

        a = int(x_cons_data.shape[0] / 10)
        b = int(hx_cas_data.shape[0] / 10)
        c = int(lx_cas_data.shape[0] / 10)

        s = cross_validation_split
        # 9 folds for training
        train = x_cons_data[:a * s]
        train = np.concatenate((train, x_cons_data[a * (s + 1):]), axis=0)

        d = int((9 * a) / (9 * (b + c)))
        d = max(1, d)
        print(d)
        total = a + (b+c)*d
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

        # random.seed(0)
        # random.shuffle(train)
        # random.shuffle(val_all)
        # random.shuffle(val_low)
        # random.shuffle(val_high)

        train_dataset = Vanilla_Dataset(train)
        val_all_dataset = Vanilla_Dataset(val_all)
        val_low_dataset = Vanilla_Dataset(val_low)
        val_high_dataset = Vanilla_Dataset(val_high)
        self.dataset = (train_dataset, val_all_dataset, val_low_dataset, val_high_dataset)
        end = time.time()
        print('total time to load data: {} secs'.format(end - start))
        # samples = prepare_data()
        # self.dataset = TensorDataset(*samples)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers, dsc_cv=True)


class Vanilla_Dataset(Dataset):
    """ Implementation of Dataset class for the synthetic dataset. """

    def __init__(self, samples):
        self.samples = torch.tensor(samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]
