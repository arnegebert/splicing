import time

import numpy as np
from torch.utils.data import Dataset
import torch

from base import BaseDataLoader

class Vanilla_DataLoader(BaseDataLoader):
    """
    PSI data loading demo using BaseDataLoader
    """

    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True,
                 classification=True, classification_treshold=0.99, cross_validation_split=0):
        self.data_dir = data_dir
        start = time.time()
        print(f'starting loading of data')
        self.samples = []
        self.class_threshold = classification_treshold
        self.cross_validation_split = cross_validation_split

        if data_dir:
            x_cons_data = np.load(f'{data_dir}/cons.npy')
            hx_cas_data = np.load(f'{data_dir}/high.npy')
            lx_cas_data = np.load(f'{data_dir}/low.npy')
        else:
            raise Exception('No data directories given!')

        self.dataset = self.cross_validation(x_cons_data, lx_cas_data, hx_cas_data)
        end = time.time()
        print('total time to load data: {} secs'.format(end - start))
        # samples = prepare_data()
        # self.dataset = TensorDataset(*samples)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers, dsc_cv=True)

    def cross_validation(self, cons, low, high):
        cons[:, 280, 3] = (cons[:, 280, 3] >= self.class_threshold).astype(np.float32)
        high[:, 280, 3] = (high[:, 280, 3] >= self.class_threshold).astype(np.float32)
        low[:, 280, 3] = (low[:, 280, 3] >= self.class_threshold).astype(np.float32)

        cons_fold_len = int(cons.shape[0] / 10)
        high_fold_len = int(high.shape[0] / 10)
        low_fold_len = int(low.shape[0] / 10)

        # 1 fold for validation & early stopping
        fold_val = 0
        fold_test = 1

        cons_to_alternative_ratio = int(cons_fold_len/(high_fold_len + low_fold_len))
        # avoid it being rounded down to 0
        cons_to_alternative_ratio = max(1, cons_to_alternative_ratio)
        print(f'cons_to_alternative_ratio: {cons_to_alternative_ratio}')
        total = cons_fold_len + (high_fold_len + low_fold_len) * cons_to_alternative_ratio
        cons_perc = cons_fold_len / total
        print(f'Percentage of consecutive data: {cons_perc}')
        if cons_perc > 0.6 or cons_perc < 0.4:
            raise Exception('Unbalanced dataset')

        # 9 folds for training
        train = cons[:cons_fold_len * fold_val]
        train = np.concatenate((train, cons[cons_fold_len * (fold_val + 1):]), axis=0)
        for _ in range(cons_to_alternative_ratio):
            train = np.concatenate((train, high[:high_fold_len * fold_val]), axis=0)
            train = np.concatenate((train, high[high_fold_len * (fold_val + 1):]), axis=0)

            train = np.concatenate((train, low[:low_fold_len * fold_val]), axis=0)
            train = np.concatenate((train, low[low_fold_len * (fold_val + 1):]), axis=0)

        np.random.seed(0)
        np.random.shuffle(train)

        val_high = np.concatenate((high[high_fold_len * fold_val:high_fold_len * (fold_val + 1)], cons[cons_fold_len * fold_val:cons_fold_len * (fold_val + 1)]), axis=0)
        val_low = np.concatenate((low[low_fold_len * fold_val:low_fold_len * (fold_val + 1)], cons[cons_fold_len * fold_val:cons_fold_len * (fold_val + 1)]), axis=0)
        val_all = np.concatenate((val_high, low[low_fold_len * fold_val:low_fold_len * (fold_val + 1)]), axis=0)

        print(f'Size training dataset: {len(train)}')
        print(f'Size mixed validation dataset: {len(val_all)}')
        print(f'Size low inclusion validation dataset: {len(val_low)}')
        print(f'Size high inclusion validation dataset: {len(val_high)}')

        train_dataset = Vanilla_Dataset(train)
        val_all_dataset = Vanilla_Dataset(val_all)
        val_low_dataset = Vanilla_Dataset(val_low)
        val_high_dataset = Vanilla_Dataset(val_high)
        return train_dataset, val_all_dataset, val_low_dataset, val_high_dataset

class Vanilla_Dataset(Dataset):
    """ Implementation of Dataset class for the synthetic dataset. """

    def __init__(self, samples):
        self.samples = torch.tensor(samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]