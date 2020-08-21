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
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.1, num_workers=1, training=True,
                 classification=True, classification_treshold=0.95, cross_validation_seed=0, embedded=False):
        self.data_dir = data_dir
        self.classification = classification
        self.class_threshold = classification_treshold
        self.cross_validation_seed = cross_validation_seed
        self.embedded = embedded
        start = time.time()
        print(f'starting loading of data')
        self.samples = []

        if data_dir:
            raise Exception('reeeeeeeeeeeeeeeeeeeee')
        else:
            cons = np.load('data/iPSC/exon/cons.npy')
            low = np.load('data/iPSC/exon/low_bezi1.npy')
            high = np.load('data/iPSC/exon/high_bezi1.npy')
            self.apply_classification_treshold(cons, low, high)

            diff_lib_cons = np.load('data/iPSC/exon/cons.npy')
            diff_lib_low = np.load('data/iPSC/exon/low_bezi2.npy')
            diff_lib_high = np.load('data/iPSC/exon/high_bezi2.npy')
            self.apply_classification_treshold(diff_lib_cons, diff_lib_low, diff_lib_high)

            diff_indv_cons = np.load('data/iPSC/exon/cons.npy')
            diff_indv_low = np.load('data/iPSC/exon/low_lexy2.npy')
            diff_indv_high = np.load('data/iPSC/exon/high_lexy2.npy')
            self.apply_classification_treshold(diff_indv_cons, diff_indv_low, diff_indv_high)

            diff_tissue_cons = np.load('data/hipsci_majiq/exon/cons.npy')
            diff_tissue_low = np.load('data/hipsci_majiq/exon/low.npy')
            diff_tissue_high = np.load('data/hipsci_majiq/exon/high.npy')
            self.apply_classification_treshold(diff_tissue_cons, diff_tissue_low, diff_tissue_high)

        self.dataset_same = (cons, low, high)#self.get_train_test_and_val_sets(cons, low, high)
        dataset_diff_lib = self.get_train_test_and_val_sets(diff_lib_cons, diff_lib_low, diff_lib_high)
        dataset_diff_indv = self.get_train_test_and_val_sets(diff_indv_cons, diff_indv_low, diff_indv_high)
        dataset_diff_tissue = self.get_train_test_and_val_sets(diff_tissue_cons, diff_tissue_low, diff_tissue_high)
        # 1:4 contains test sets
        test_datasets_diff_lib = dataset_diff_lib[1:4]
        test_datasets_diff_indv = dataset_diff_indv[1:4]
        test_datasets_diff_tissue = dataset_diff_tissue[1:4]

        # flatten dataset elements
        extra_test_datasets = [sample for dataset in
                              [test_datasets_diff_lib, test_datasets_diff_indv, test_datasets_diff_tissue]
                              for sample in dataset]

        end = time.time()

        print('total time to load data: {} secs'.format(end - start))
        super().__init__(self.dataset_same, batch_size, shuffle, validation_split, num_workers,
                         extra_test_datasets=extra_test_datasets)

    def apply_classification_treshold(self, cons, low, high):
        if self.classification and not self.embedded:
            cons[:, 280, 3] = (cons[:, 280, 3] >= self.class_threshold).astype(np.float32)
            low[:, 280, 3] = (low[:, 280, 3] >= self.class_threshold).astype(np.float32)
            high[:, 280, 3] = (high[:, 280, 3] >= self.class_threshold).astype(np.float32)

    # # overriding split_validation for extra functionality
    # def split_validation(self):
    #     pass


