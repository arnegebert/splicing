import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np

class BaseDataLoader(DataLoader):
    """
    Base class for all data loaders
    """
    def __init__(self, train, test, val, batch_size, shuffle, validation_split, num_workers,
                 extra_test_datasets=None, drop_last=False,
                 data_split=True, cross_validation_seed=0):
        assert data_split, "Handling when data is not split into cons/low/high data currently not implemented"
        self.train, self.val = train, val
        if data_split:
            self.test_all, self.test_low, self.test_high = test
        self.validation_split = validation_split
        self.cross_validation_seed = cross_validation_seed
        self.extra_test = extra_test_datasets

        self.init_kwargs = {
            'batch_size': batch_size,
            'shuffle': shuffle,
            'num_workers': num_workers,
            'drop_last': drop_last
        }

        super().__init__(dataset=self.train, **self.init_kwargs)

    def get_valid_and_test_loaders(self):
        if not self.extra_test:
            return DataLoader(dataset=self.test_all,  **self.init_kwargs),\
                   DataLoader(dataset=self.test_low, **self.init_kwargs),\
                   DataLoader(dataset=self.test_high,  **self.init_kwargs),\
                   DataLoader(dataset=self.val,  **self.init_kwargs)#,\
                   # DataLoader(dataset=self.test_low, **self.init_kwargs),\
                   # DataLoader(dataset=self.test_high,  **self.init_kwargs)
        else:
            # three test and one val set which are always there
            regular_test_and_val_sets = [DataLoader(dataset=self.test_all,  **self.init_kwargs),
                                         DataLoader(dataset=self.test_low, **self.init_kwargs),
                                         DataLoader(dataset=self.test_high,  **self.init_kwargs),
                                         DataLoader(dataset=self.val,  **self.init_kwargs)]
            # 9 other test sets which I have added for cross-condition performance
            if len(self.extra_test) != 9: raise Exception('Unexpected number of extra test sets')
            extra_test_sets = []
            for extra_test_set in self.extra_test:
                extra_test_sets.append(DataLoader(dataset=extra_test_set,  **self.init_kwargs))
            return regular_test_and_val_sets, extra_test_sets

    def apply_classification_threshold(self, cons, low, high, embedded=False, threshold=0.99):
        if not embedded:
            cons[:, 280, 3] = (cons[:, 280, 3] >= threshold).astype(np.float32)
            low[:, 280, 3] = (low[:, 280, 3] >= threshold).astype(np.float32)
            high[:, 280, 3] = (high[:, 280, 3] >= threshold).astype(np.float32)
            high_cons, high_non_cons = high[high[:, 280, 3] == 1], high[high[:, 280, 3] == 0]
        else:
            cons[:, 2, 3] = (cons[:, 2, 3] >= threshold).astype(np.float32)
            low[:, 2, 3] = (low[:, 2, 3] >= threshold).astype(np.float32)
            high[:, 2, 3] = (high[:, 2, 3] >= threshold).astype(np.float32)
            high_cons, high_non_cons = high[high[:, 2, 3] == 1], high[high[:, 2, 3] == 0]
        # data is initially split low/high/cons PSI, but after classification some high PSI
        # might be classified as constitutive -> adjust split
        cons = np.concatenate((cons, high_cons), axis=0)
        high = high_non_cons
        return cons, low, high


    def get_train_test_and_val_sets(self, cons, low, high, folds=10):
        cons_fold_len = int(cons.shape[0] / folds)
        high_fold_len = int(high.shape[0] / folds)
        low_fold_len = int(low.shape[0] / folds)

        # shuffling to avoid any biases from data generation
        np.random.seed(0)
        np.random.shuffle(cons)
        np.random.shuffle(low)
        np.random.shuffle(high)
        # 1 fold for validation & early stopping
        fold_val = self.cross_validation_seed
        # 1 fold for testing
        fold_test = fold_val + 1

        cons_to_alternative_ratio = int(cons_fold_len / (high_fold_len + low_fold_len))
        # avoid ratio being rounded down to 0
        cons_to_alternative_ratio = max(1, cons_to_alternative_ratio)
        print(f'cons_to_alternative_ratio: {cons_to_alternative_ratio}')
        total = cons_fold_len + (high_fold_len + low_fold_len) * cons_to_alternative_ratio
        cons_perc = cons_fold_len / total
        print(f'Percentage of consecutive data: {cons_perc}')
        # if cons_perc > 0.66 or cons_perc < 0.34:
        #     raise Exception(f'Unbalanced dataset')

        # 9 folds for training
        train = cons[:cons_fold_len * fold_val]
        train = np.concatenate((train, cons[cons_fold_len * (fold_test + 1):]), axis=0)
        # rebalancing of dataset
        for _ in range(cons_to_alternative_ratio):
            train = np.concatenate((train, high[:high_fold_len * fold_val]), axis=0)
            train = np.concatenate((train, high[high_fold_len * (fold_test + 1):]), axis=0)

            train = np.concatenate((train, low[:low_fold_len * fold_val]), axis=0)
            train = np.concatenate((train, low[low_fold_len * (fold_test + 1):]), axis=0)


        val_high = np.concatenate((high[high_fold_len * fold_val:high_fold_len * (fold_val + 1)],
                                   cons[cons_fold_len * fold_val:cons_fold_len * (fold_val + 1)]), axis=0)
        val_low = np.concatenate((low[low_fold_len * fold_val:low_fold_len * (fold_val + 1)],
                                  cons[cons_fold_len * fold_val:cons_fold_len * (fold_val + 1)]), axis=0)
        val_all = np.concatenate((val_high, low[low_fold_len * fold_val:low_fold_len * (fold_val + 1)]), axis=0)


        # test_high = test_low = test_all = cons[:1]

        # test_high = test_low = test_all = cons[cons_fold_len * fold_test:cons_fold_len * (fold_test + 1)]

        test_high = np.concatenate((high[high_fold_len * fold_test:high_fold_len * (fold_test + 1)],
                                    cons[cons_fold_len * fold_test:cons_fold_len * (fold_test + 1)]), axis=0)
        test_low = np.concatenate((low[low_fold_len * fold_test:low_fold_len * (fold_test + 1)],
                                   cons[cons_fold_len * fold_test:cons_fold_len * (fold_test + 1)]), axis=0)
        test_all = np.concatenate((test_high, low[low_fold_len * fold_test:low_fold_len * (fold_test + 1)]), axis=0)

        print(f'Size training dataset: {len(train)}')
        print(f'Proportions (after balancing):')
        print(f'cons: {cons_fold_len/total:.3f}%')
        print(f'low: {low_fold_len*cons_to_alternative_ratio/total:.3f}%')
        print(f'high: {high_fold_len*cons_to_alternative_ratio/total:.3f}%')

        train_dataset = VanillaDataset(train)
        val_all_dataset, val_low_dataset, val_high_dataset = VanillaDataset(val_all), VanillaDataset(val_low), \
                                                             VanillaDataset(val_high)
        test_all_dataset, test_low_dataset, test_high_dataset = VanillaDataset(test_all), VanillaDataset(test_low), \
                                                                VanillaDataset(test_high)
        return train_dataset, test_all_dataset, test_low_dataset, test_high_dataset, \
               val_all_dataset  # , val_low_dataset, val_high_dataset

    @staticmethod
    def construct_dataset(data):
        return VanillaDataset(data)

class VanillaDataset(Dataset):
    """ Simple Implementation of Dataset class. """

    def __init__(self, samples):
        self.samples = torch.tensor(samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]