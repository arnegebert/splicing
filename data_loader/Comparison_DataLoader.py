import time

import numpy as np
from torch.utils.data import Dataset
import torch
from base import BaseDataLoader

class Comparison_DataLoader(BaseDataLoader):
    """
    Implements all three use cases for HIPSCI NotNeuron:
    1. Train on lib1, test on lib1
    2. Train on lib1, test on lib2
    3. Train on lib1, test on lib from other individual
    4. Train on lib1, test on lib from other individual other tissue
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True,
                 classification=True, classification_treshold=0.95, cross_validation_split=0, embedded=False):
        self.data_dir = data_dir
        self.classification = classification
        self.class_threshold = classification_treshold
        self.cross_validation_split = cross_validation_split
        self.embedded = embedded
        start = time.time()
        print(f'starting loading of data')
        self.samples = []

        if data_dir:
            raise Exception('reeeeeeeeeeeeeeeeeeeee')
        else:
            cons = np.load('data/iPSC/exon/cons.npy')
            high = np.load('data/iPSC/exon/high_bezi1.npy')
            low = np.load('data/iPSC/exon/low_bezi1.npy')

            diff_lib_cons = np.load('data/iPSC/exon/cons.npy')
            diff_lib_high = np.load('data/iPSC/exon/high_bezi2.npy')
            diff_lib_low = np.load('data/iPSC/exon/low_bezi2.npy')

            diff_indv_cons = np.load('data/iPSC/exon/cons.npy')
            diff_indv_high = np.load('data/iPSC/exon/high_lexy2.npy')
            diff_indv_low = np.load('data/iPSC/exon/low_lexy2.npy')

            diff_tissue_cons = np.load('data/hipsci_majiq/exon/cons.npy')
            diff_tissue_high = np.load('data/hipsci_majiq/exon/high.npy')
            diff_tissue_low = np.load('data/hipsci_majiq/exon/low.npy')

        self.dataset_same = self.cross_validation(cons, low, high)
        #train_dataset, val_all_dataset, val_low_dataset, val_high_dataset = self.cross_validation(cons, low, high)
        dataset_diff_lib = self.cross_validation(diff_lib_cons, diff_lib_low, diff_lib_high)
        dataset_diff_indv = self.cross_validation(diff_indv_cons, diff_indv_low, diff_indv_high)
        dataset_diff_tissue = self.cross_validation(diff_tissue_cons, diff_tissue_low, diff_tissue_high)
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

    def cross_validation(self, cons, low, high, folds=10):
        if not self.embedded:
            cons[:, 280, 3] = (cons[:, 280, 3] >= self.class_threshold).astype(np.float32)
            high[:, 280, 3] = (high[:, 280, 3] >= self.class_threshold).astype(np.float32)
            low[:, 280, 3] = (low[:, 280, 3] >= self.class_threshold).astype(np.float32)

        cons_fold_len = int(cons.shape[0] / folds)
        high_fold_len = int(high.shape[0] / folds)
        low_fold_len = int(low.shape[0] / folds)

        # 1 fold for validation & early stopping
        fold_val = 0
        fold_test = 1

        cons_to_alternative_ratio = int(cons_fold_len / (high_fold_len + low_fold_len))
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
        train = np.concatenate((train, cons[cons_fold_len * (fold_test + 1):]), axis=0)
        for _ in range(cons_to_alternative_ratio):
            train = np.concatenate((train, high[:high_fold_len * fold_val]), axis=0)
            train = np.concatenate((train, high[high_fold_len * (fold_test + 1):]), axis=0)

            train = np.concatenate((train, low[:low_fold_len * fold_val]), axis=0)
            train = np.concatenate((train, low[low_fold_len * (fold_test + 1):]), axis=0)

        np.random.seed(0)
        np.random.shuffle(train)

        val_high = np.concatenate((high[high_fold_len * fold_val:high_fold_len * (fold_val + 1)],
                                   cons[cons_fold_len * fold_val:cons_fold_len * (fold_val + 1)]), axis=0)
        val_low = np.concatenate((low[low_fold_len * fold_val:low_fold_len * (fold_val + 1)],
                                  cons[cons_fold_len * fold_val:cons_fold_len * (fold_val + 1)]), axis=0)
        val_all = np.concatenate((val_high, low[low_fold_len * fold_val:low_fold_len * (fold_val + 1)]), axis=0)

        test_high = np.concatenate((high[high_fold_len * fold_test:high_fold_len * (fold_test + 1)],
                                    cons[cons_fold_len * fold_test:cons_fold_len * (fold_test + 1)]), axis=0)
        test_low = np.concatenate((low[low_fold_len * fold_test:low_fold_len * (fold_test + 1)],
                                   cons[cons_fold_len * fold_test:cons_fold_len * (fold_test + 1)]), axis=0)
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
        return train_dataset, test_all_dataset, test_low_dataset, test_high_dataset, \
               val_all_dataset  # , val_low_dataset, val_high_dataset

class VanillaDataset(Dataset):
    """ Implementation of Dataset class for the synthetic dataset. """

    def __init__(self, samples):
        self.samples = torch.tensor(samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]