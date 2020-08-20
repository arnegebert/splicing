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
