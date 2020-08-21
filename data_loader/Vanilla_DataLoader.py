import time

import numpy as np
from torch.utils.data import Dataset
import torch

from base import BaseDataLoader

class Vanilla_DataLoader(BaseDataLoader):
    """
    DataLoader for loading data in the standard "Vanilla" format
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
            prefix = "embedded_" if embedded else ""
            x_cons_data = np.load(f'{data_dir}/{prefix}cons.npy')
            hx_cas_data = np.load(f'{data_dir}/{prefix}high.npy')
            lx_cas_data = np.load(f'{data_dir}/{prefix}low.npy')
        else:
            raise Exception('No data directories given!')

        if classification:
            self.apply_classification_threshold(x_cons_data, lx_cas_data, hx_cas_data, embedded=embedded,
                                                threshold=classification_threshold)

        # maybe todo: make class know about cross-validation (do this later though and ideally only the derived class knows)
        # but maybe baseloader can know about it too since it would be kinda silly otherwise and I do need the functionality i nbtoh
        self.current_cv_seed = cross_validation_seed
        end = time.time()
        print('total time to load data: {} secs'.format(end - start))
        super().__init__((x_cons_data, hx_cas_data, lx_cas_data),
                         batch_size, shuffle, validation_split, num_workers, cross_validation_seed)

    def get_train_test_and_val(self):
        pass
