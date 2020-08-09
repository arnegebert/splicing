from base import BaseDataLoader
import time

import numpy as np
import torch
from torch.utils.data import Dataset

from base import BaseDataLoader


class GTEx_EmbeddedDataLoader(BaseDataLoader):
    """
    Simply loads embedded data from one file and let's it be split by BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True,
                 cross_validation_split=0):
        # todo: cross_validation_split parameter ignored
        self.data_dir = data_dir
        self.dataset = GTEx_EmbeddedDataset(path=data_dir)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

class GTEx_EmbeddedDataset(Dataset):

    def __init__(self, path, transform=None):
        startt = time.time()
        print(f'starting loading of data')
        self.samples = np.load(path)
        np.random.seed(0)
        np.random.shuffle(self.samples)

        self.samples = torch.from_numpy(self.samples)

        endt = time.time()
        print(f'Took {endt-startt} to load data')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]