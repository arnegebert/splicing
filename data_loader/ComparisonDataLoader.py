import time

import numpy as np

from base import BaseDataLoader


class ComparisonDataLoader(BaseDataLoader):
    """
    Implements all three use cases for HIPSCI NotNeuron:
    1. Train on lib1, test on lib1
    2. Train on lib1, test on lib2
    3. Train on lib1, test on lib from other individual
    4. Train on lib1, test on lib from other individual other tissue
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.1, num_workers=1,
                 classification=True, classification_threshold=0.99, cross_validation_seed=0, embedded=False):
        self.data_dir = data_dir
        self.classification = classification
        self.class_threshold = classification_threshold
        self.cross_validation_seed = cross_validation_seed
        self.embedded = embedded
        start = time.time()
        print(f'starting loading of data')

        if data_dir:
            raise Exception('Data directories for this data loader are preset and can\'t be overwritten.')
        else:
            prefix = "embedded_" if embedded else ""

            cons = np.load(f'data/iPSC/exon/{prefix}cons.npy')
            low = np.load(f'data/iPSC/exon/{prefix}low_bezi1.npy')
            high = np.load(f'data/iPSC/exon/{prefix}high_bezi1.npy')

            diff_lib_cons = np.load(f'data/iPSC/exon/{prefix}cons.npy')
            diff_lib_low = np.load(f'data/iPSC/exon/{prefix}low_bezi2.npy')
            diff_lib_high = np.load(f'data/iPSC/exon/{prefix}high_bezi2.npy')

            diff_indv_cons = np.load(f'data/iPSC/exon/{prefix}cons.npy')
            diff_indv_low = np.load(f'data/iPSC/exon/{prefix}low_lexy2.npy')
            diff_indv_high = np.load(f'data/iPSC/exon/{prefix}high_lexy2.npy')

            diff_tissue_cons = np.load(f'data/hipsci_majiq/exon/{prefix}cons.npy')
            diff_tissue_low = np.load(f'data/hipsci_majiq/exon/{prefix}low.npy')
            diff_tissue_high = np.load(f'data/hipsci_majiq/exon/{prefix}high.npy')

        if self.classification:
            cons, low, high = self.apply_classification_threshold(cons, low, high,
                                                embedded=embedded, threshold=classification_threshold)
            diff_lib_cons, diff_lib_low, diff_lib_high = \
                self.apply_classification_threshold(diff_lib_cons, diff_lib_low, diff_lib_high,
                                                embedded=embedded, threshold=classification_threshold)
            diff_indv_cons, diff_indv_low, diff_indv_high = \
                self.apply_classification_threshold(diff_indv_cons, diff_indv_low, diff_indv_high,
                                                embedded=embedded, threshold=classification_threshold)
            diff_tissue_cons, diff_tissue_low, diff_tissue_high = \
                self.apply_classification_threshold(diff_tissue_cons, diff_tissue_low, diff_tissue_high,
                                                embedded=embedded, threshold=classification_threshold)

        train, test_all, test_low, test_high, val = self.get_train_test_and_val_sets(cons, low, high)

        filter_hashset = self.construct_hashset(train, val)
        diff_lib_cons, diff_lib_low, diff_lib_high = self.filter_for_data_leak(filter_hashset, diff_lib_cons,
                                                                              diff_lib_low, diff_lib_high)
        diff_indv_cons, diff_indv_low, diff_indv_high = self.filter_for_data_leak(filter_hashset, diff_indv_cons,
                                                                              diff_indv_low, diff_indv_high)
        diff_tissue_cons, diff_tissue_low, diff_tissue_high = self.filter_for_data_leak(filter_hashset, diff_tissue_cons,
                                                                              diff_tissue_low, diff_tissue_high)
        test_datasets_diff_lib = self._construct_all_low_and_high_datasets(diff_lib_cons, diff_lib_low, diff_lib_high)
        test_datasets_diff_indv = self._construct_all_low_and_high_datasets(diff_indv_cons, diff_indv_low, diff_indv_high)
        test_datasets_diff_tissue = self._construct_all_low_and_high_datasets(diff_tissue_cons, diff_tissue_low, diff_tissue_high)

        # flatten dataset elements
        extra_test_datasets = [sample for dataset in
                              [test_datasets_diff_lib, test_datasets_diff_indv, test_datasets_diff_tissue]
                              for sample in dataset]


        super().__init__(train, (test_all, test_low, test_high), val, batch_size, shuffle, validation_split, num_workers,
                         cross_validation_seed=cross_validation_seed, extra_test_datasets=extra_test_datasets)
        end = time.time()
        print('total time to load data: {} secs'.format(end - start))

    # overriding get_valid_and_test_loaders for extra functionality;
    # would be cleaner since it means that base dataloader no longer needs to know about this class
    # def get_valid_and_test_loaders(self):
    #     pass

    @staticmethod
    def _construct_all_low_and_high_datasets(cons, low, high):
        np.random.seed(0)
        np.random.shuffle(cons)
        np.random.shuffle(low)
        np.random.shuffle(high)

        all_d = np.concatenate((cons, low, high), axis=0)
        low_d = np.concatenate((cons, low), axis=0)
        high_d = np.concatenate((cons, high), axis=0)
        all_set, low_set, high_set = ComparisonDataLoader.construct_dataset(all_d), \
                                     ComparisonDataLoader.construct_dataset(low_d), \
                                     ComparisonDataLoader.construct_dataset(high_d)
        return all_set, low_set, high_set

    @staticmethod
    def construct_hashset(train, val):
        train_np, val_np =  train.samples.numpy(), val.samples.numpy()
        train_and_val_samples = set()
        for sample in list(train_np[:, :281]):
            train_and_val_samples.add(sample.tostring())
        return train_and_val_samples

    @staticmethod
    def filter_for_data_leak(hashset, possibilities_cons, possibilities_low, possibilities_high):
        cons_filtered = ComparisonDataLoader.apply_filter_for_data_leak(hashset, possibilities_cons)
        low_filtered = ComparisonDataLoader.apply_filter_for_data_leak(hashset, possibilities_low)
        high_filtered = ComparisonDataLoader.apply_filter_for_data_leak(hashset, possibilities_high)
        return cons_filtered, low_filtered, high_filtered

    @staticmethod
    def apply_filter_for_data_leak(hashset, possibilities):
        poss_filtered = []
        for sample in list(possibilities[:, :281]):
            if sample.tostring() not in hashset:
                poss_filtered.append(sample)
        # 35954 / 5827 w/ only sequence information
        # 36179 / 6413 w/ length information too
        return poss_filtered


