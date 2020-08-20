import time

import numpy as np
from torch.utils.data import Dataset
import torch

from base import BaseDataLoader

class Vanilla_DataLoader(BaseDataLoader):
    """
    PSI data loading demo using BaseDataLoader
    """

    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.1, num_workers=1,
                 classification=True, classification_treshold=0.99, cross_validation_seed=0, embedded=False):
        self.data_dir = data_dir
        start = time.time()
        print(f'starting loading of data')
        self.class_threshold = classification_treshold
        self.cross_validation_seed = cross_validation_seed
        self.embedded = embedded
        self.classification = classification
        if data_dir:
            prefix = "embedded_" if embedded else ""
            x_cons_data = np.load(f'{data_dir}/{prefix}cons.npy')
            hx_cas_data = np.load(f'{data_dir}/{prefix}high.npy')
            lx_cas_data = np.load(f'{data_dir}/{prefix}low.npy')
        else:
            raise Exception('No data directories given!')

        # todo; make not dependent on self.embedded
        if self.classification and not self.embedded:
            x_cons_data[:, 280, 3] = (x_cons_data[:, 280, 3] >= self.class_threshold).astype(np.float32)
            hx_cas_data[:, 280, 3] = (hx_cas_data[:, 280, 3] >= self.class_threshold).astype(np.float32)
            lx_cas_data[:, 280, 3] = (lx_cas_data[:, 280, 3] >= self.class_threshold).astype(np.float32)

        # todo: make class know about cross-validation (do this later though and ideally only the derived class knows)
        # but maybe baseloader can know about it too since it would be kinda silly otherwise and I do need the functionality i nbtoh
        self.current_cv_seed = cross_validation_seed
        end = time.time()
        print('total time to load data: {} secs'.format(end - start))
        super().__init__((x_cons_data, hx_cas_data, lx_cas_data),
                         batch_size, shuffle, validation_split, num_workers, cross_validation_seed)

    def get_train_test_and_val(self):
        pass

    def cross_validation(self, cons, low, high, folds=10):
        # todo; make not dependent on self.embedded
        if self.classification and not self.embedded:
            cons[:, 280, 3] = (cons[:, 280, 3] >= self.class_threshold).astype(np.float32)
            high[:, 280, 3] = (high[:, 280, 3] >= self.class_threshold).astype(np.float32)
            low[:, 280, 3] = (low[:, 280, 3] >= self.class_threshold).astype(np.float32)

        cons_fold_len = int(cons.shape[0] / folds)
        high_fold_len = int(high.shape[0] / folds)
        low_fold_len = int(low.shape[0] / folds)

        # 1 fold for validation & early stopping
        fold_val = self.cross_validation_seed
        fold_test = fold_val + 1

        cons_to_alternative_ratio = int(cons_fold_len/(high_fold_len + low_fold_len))
        # avoid it being rounded down to 0
        cons_to_alternative_ratio = max(1, cons_to_alternative_ratio)
        print(f'cons_to_alternative_ratio: {cons_to_alternative_ratio}')
        total = cons_fold_len + (high_fold_len + low_fold_len) * cons_to_alternative_ratio
        cons_perc = cons_fold_len / total
        print(f'Percentage of consecutive data: {cons_perc}')
        if cons_perc > 0.6 or cons_perc < 0.4:
            raise Exception(f'Unbalanced dataset')

        # 9 folds for training
        train = cons[:cons_fold_len * fold_val]
        train = np.concatenate((train, cons[cons_fold_len * (fold_test + 1):]), axis=0)
        for _ in range(cons_to_alternative_ratio):
            train = np.concatenate((train, high[:high_fold_len * fold_val]), axis=0)
            train = np.concatenate((train, high[high_fold_len * (fold_test + 1):]), axis=0)

            train = np.concatenate((train, low[:low_fold_len * fold_val]), axis=0)
            train = np.concatenate((train, low[low_fold_len * (fold_test + 1):]), axis=0)

        np.random.seed(0)
        np.random.shuffle(train)

        val_high = np.concatenate((high[high_fold_len * fold_val:high_fold_len * (fold_val + 1)], cons[cons_fold_len * fold_val:cons_fold_len * (fold_val + 1)]), axis=0)
        val_low = np.concatenate((low[low_fold_len * fold_val:low_fold_len * (fold_val + 1)], cons[cons_fold_len * fold_val:cons_fold_len * (fold_val + 1)]), axis=0)
        val_all = np.concatenate((val_high, low[low_fold_len * fold_val:low_fold_len * (fold_val + 1)]), axis=0)

        test_high = np.concatenate((high[high_fold_len * fold_test:high_fold_len * (fold_test + 1)], cons[cons_fold_len * fold_test:cons_fold_len * (fold_test + 1)]), axis=0)
        test_low = np.concatenate((low[low_fold_len * fold_test:low_fold_len * (fold_test + 1)], cons[cons_fold_len * fold_test:cons_fold_len * (fold_test + 1)]), axis=0)
        test_all = np.concatenate((val_high, low[low_fold_len * fold_test:low_fold_len * (fold_test + 1)]), axis=0)

        print(f'Size training dataset: {len(train)}')
        print(f'Size mixed validation dataset: {len(val_all)}')
        print(f'Size low inclusion validation dataset: {len(val_low)}')
        print(f'Size high inclusion validation dataset: {len(val_high)}')

        train_dataset = VanillaDataset(train)
        val_all_dataset, val_low_dataset, val_high_dataset = VanillaDataset(val_all), VanillaDataset(val_low), \
                                                             VanillaDataset(val_high)
        test_all_dataset, test_low_dataset, test_high_dataset = VanillaDataset(test_all), VanillaDataset(test_low), \
                                                             VanillaDataset(test_high)
        return train_dataset, test_all_dataset, test_low_dataset, test_high_dataset,\
               val_all_dataset#, val_low_dataset, val_high_dataset


class VanillaDataset(Dataset):
    """ Implementation of Dataset class for the synthetic dataset. """

    def __init__(self, samples):
        self.samples = torch.tensor(samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]