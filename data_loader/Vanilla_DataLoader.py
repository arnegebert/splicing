import time

import numpy as np
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
            cons = np.load(f'{data_dir}/{prefix}cons.npy')
            low = np.load(f'{data_dir}/{prefix}low.npy')
            high = np.load(f'{data_dir}/{prefix}high.npy')
        else:
            raise Exception('No data directories given!')

        if classification:
            cons, low, high = self.apply_classification_threshold(cons, low, high, embedded=embedded,
                                                threshold=classification_threshold)

        # maybe todo: make class know about cross-validation (do this later though and ideally only the derived class knows)
        # but maybe baseloader can know about it too since it would be kinda silly otherwise and I do need the functionality i nbtoh
        self.current_cv_seed = cross_validation_seed
        super().__init__((cons, low, high),
                         batch_size, shuffle, validation_split, num_workers, cross_validation_seed=cross_validation_seed)
        end = time.time()
        print('total time to load data: {} secs'.format(end - start))
