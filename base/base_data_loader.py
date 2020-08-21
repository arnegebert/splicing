import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np

class BaseDataLoader(DataLoader):
    """
    Base class for all data loaders
    """
    def __init__(self, data, batch_size, shuffle, validation_split, num_workers,
                 extra_test_datasets=None, drop_last=False,
                 data_split=True, cross_validation_seed=0):
        assert data_split, "Handling when data is not split into cons/low/high data currently not implemented"
        self.validation_split = validation_split
        self.cross_validation_seed = cross_validation_seed
        self.extra_test = extra_test_datasets
        if data_split:
            cons, low, high = data
        folds = 1/validation_split
        if not folds.is_integer(): print(f'Warning: rounded down to {folds} cross-validation folds')
        self.train, self.test_all, self.test_low, self.test_high, \
            self.val_all = self.get_train_test_and_val_sets(cons, low, high)

        self.init_kwargs = {
            'batch_size': batch_size,
            'shuffle': shuffle,
            'num_workers': num_workers,
            'drop_last': drop_last
        }

        super().__init__(dataset=self.train, **self.init_kwargs)

    def get_valid_and_test_loaders(self):
        if not self.extra_test:
            # return three dataloaders here based on my validation datasets
            return DataLoader(dataset=self.test_all,  **self.init_kwargs),\
                   DataLoader(dataset=self.test_low, **self.init_kwargs),\
                   DataLoader(dataset=self.test_high,  **self.init_kwargs),\
                   DataLoader(dataset=self.val_all,  **self.init_kwargs)#,\
                   # DataLoader(dataset=self.test_low, **self.init_kwargs),\
                   # DataLoader(dataset=self.test_high,  **self.init_kwargs)
        else:
            # three test sets which are standard in CV
            regular_test_and_val_sets = [DataLoader(dataset=self.test_all,  **self.init_kwargs),
                   DataLoader(dataset=self.test_low, **self.init_kwargs),
                   DataLoader(dataset=self.test_high,  **self.init_kwargs),
                         DataLoader(dataset=self.val_all,  **self.init_kwargs)]
            # 9 other test sets which I have added
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

        # 1 fold for validation & early stopping
        fold_val = self.cross_validation_seed
        # 1 fold for testing
        fold_test = fold_val + 1

        cons_to_alternative_ratio = int(cons_fold_len / (high_fold_len + low_fold_len))
        # avoid it being rounded down to 0
        cons_to_alternative_ratio = max(1, cons_to_alternative_ratio)
        print(f'cons_to_alternative_ratio: {cons_to_alternative_ratio}')
        total = cons_fold_len + (high_fold_len + low_fold_len) * cons_to_alternative_ratio
        cons_perc = cons_fold_len / total
        print(f'Percentage of consecutive data: {cons_perc}')
        if cons_perc > 0.6 or cons_perc < 0.4:
            raise Exception(f'Unbalanced dataset')

        # 9 folds for training
        train = cons[:cons_fold_len * fold_val]
        train = np.concatenate((train, cons[cons_fold_len * (fold_test + 1):]), axis=0)
        # rebalancing of dataset
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
        print(f'Size mixed test dataset: {len(test_all)}')
        print(f'Size low inclusion test dataset: {len(test_low)}')
        print(f'Size high inclusion test dataset: {len(test_high)}')
        print(f'Size mixed validation dataset: {len(val_all)}')

        train_dataset = VanillaDataset(train)
        val_all_dataset, val_low_dataset, val_high_dataset = VanillaDataset(val_all), VanillaDataset(val_low), \
                                                             VanillaDataset(val_high)
        test_all_dataset, test_low_dataset, test_high_dataset = VanillaDataset(test_all), VanillaDataset(test_low), \
                                                                VanillaDataset(test_high)
        return train_dataset, test_all_dataset, test_low_dataset, test_high_dataset, \
               val_all_dataset  # , val_low_dataset, val_high_dataset


class VanillaDataset(Dataset):
    """ Simple Implementation of Dataset class. """

    def __init__(self, samples):
        self.samples = torch.tensor(samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]