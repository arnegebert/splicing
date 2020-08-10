import time

import numpy as np
from torch.utils.data import Dataset

from base import BaseDataLoader


class HEXEvent2Vanilla_DataLoader(BaseDataLoader):
    """
    Bit different than the standard I use for data loading (so can't directly replace with Vanilla_DataLoader)
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True,
                 cross_validation_split=0, classification_treshold=0.99):
        self.data_dir = data_dir
        start = time.time()
        print(f'starting loading of data')
        self.samples = []

        x_cons_data = np.load('data/hexevent/cons.npy')
        hx_cas_data = np.load('data/hexevent/high.npy')
        lx_cas_data = np.load('data/hexevent/low.npy')

        x_cons_data[:,-1,4] = 1
        x_cons_data = convert_hexevent_to_vanilla_format(x_cons_data, classification_treshold)
        hx_cas_data = convert_hexevent_to_vanilla_format(hx_cas_data, classification_treshold)
        lx_cas_data = convert_hexevent_to_vanilla_format(lx_cas_data, classification_treshold)

        a = int(x_cons_data.shape[0] / 10)
        b = int(hx_cas_data.shape[0] / 10)
        c = int(lx_cas_data.shape[0] / 10)

        s = cross_validation_split
        # 9 folds for training
        train = x_cons_data[:a * s]
        train = np.concatenate((train, x_cons_data[a * (s + 1):]), axis=0)

        d = int((9 * a) / (9 * (b + c)))
        d = max(1, d)
        total = a + (b + c) * d
        cons_perc = a / total
        print(f'Percentage of consecutive data: {cons_perc}')
        if cons_perc > 0.6 or cons_perc < 0.4:
            raise Exception('Unbalanced dataset')
        print(d)
        for i in range(d):
            train = np.concatenate((train, hx_cas_data[:b * s]), axis=0)
            train = np.concatenate((train, hx_cas_data[b * (s + 1):]), axis=0)

            train = np.concatenate((train, lx_cas_data[:c * s]), axis=0)
            train = np.concatenate((train, lx_cas_data[c * (s + 1):]), axis=0)

        np.random.seed(0)
        np.random.shuffle(train)

        # 1 fold for testing

        htest = np.concatenate((hx_cas_data[b * s:b * (s + 1)], x_cons_data[a * s:a * (s + 1)]), axis=0)
        lt = np.concatenate((lx_cas_data[c * s:c * (s + 1)], x_cons_data[a * s:a * (s + 1)]), axis=0)

        test = htest
        test = np.concatenate((test, lx_cas_data[c * s:c * (s + 1)]), axis=0)

        cons_test = x_cons_data[a * s:a * (s + 1)]
        cas_test = np.concatenate((lx_cas_data[c * s:c * (s + 1)], hx_cas_data[b * s:b * (s + 1)]))

        train = train
        # cons + low + high
        val_all = test
        # cons + low
        val_low = lt
        # cons + high
        val_high = htest
        train_dataset = DSCDataset(train)
        val_all_dataset = DSCDataset(val_all)
        val_low_dataset = DSCDataset(val_low)
        val_high_dataset = DSCDataset(val_high)
        self.dataset = (train_dataset, val_all_dataset, val_low_dataset, val_high_dataset)
        end = time.time()
        print('total time to load data: {} secs'.format(end - start))
        # samples = prepare_data()
        # self.dataset = TensorDataset(*samples)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers, dsc_cv=True)

# Constant values taken in reference from
# https://github.com/louadi/DSC/blob/master/training%20notebooks/cons_vs_es.ipynb
# Don't blame me ¯\_(ツ)_/¯
def convert_hexevent_to_vanilla_format(array, treshold):
    lifehack = 500000
    label = array[:lifehack, -1, 4] > treshold
    start_seq, end_seq = array[:lifehack, :140, :4], array[:lifehack, 141:281, :4]
    start_and_end = np.concatenate((start_seq, end_seq), axis=1)
    lens = array[:lifehack, -1, 0:3]
    lens_and_psi = np.concatenate((lens, label.reshape(-1, 1)), axis=1).reshape(-1, 1, 4)
    samples = np.concatenate((start_and_end, lens_and_psi), axis=1).astype(np.float32)
    return samples

class DSCDataset(Dataset):
    """ Implementation of Dataset class for the synthetic dataset. """

    def __init__(self, samples):
        # random.seed(0)
        # random.shuffle(samples)
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]