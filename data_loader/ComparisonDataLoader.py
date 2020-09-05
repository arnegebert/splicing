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
                 classification=True, classification_threshold=0.95, cross_validation_seed=0, embedded=False):
        self.data_dir = data_dir
        self.classification = classification
        self.class_threshold = classification_threshold
        self.cross_validation_seed = cross_validation_seed
        self.embedded = embedded
        start = time.time()
        print(f'starting loading of data')

        if data_dir:
            raise Exception('reeeeeeeeeeeeeeeeeeeee')
        else:
            prefix = "embedded_" if embedded else ""
            cons = np.load(f'data/iPSC/exon/{prefix}cons.npy')
            low = np.load(f'data/iPSC/exon/{prefix}low_bezi2.npy')
            high = np.load(f'data/iPSC/exon/{prefix}high_bezi2.npy')

            diff_lib_cons = np.load(f'data/iPSC/exon/{prefix}cons.npy')
            diff_lib_low = np.load(f'data/iPSC/exon/{prefix}low_bezi1.npy')
            diff_lib_high = np.load(f'data/iPSC/exon/{prefix}high_bezi1.npy')

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


        super().__init__((cons, low, high), batch_size, shuffle, validation_split, num_workers,
                         cross_validation_seed=cross_validation_seed,
                         extra_test_datasets=extra_test_datasets)
        end = time.time()
        print('total time to load data: {} secs'.format(end - start))

    # overriding get_valid_and_test_loaders for extra functionality;
    # would be cleaner since it means that base dataloader no longer needs to know about this class
    # def get_valid_and_test_loaders(self):
    #     pass


