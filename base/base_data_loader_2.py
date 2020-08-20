import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataloader import default_collate
import numpy as np

class BaseDataLoader(DataLoader):
    """
    Base class for all data loaders
    """
    def __init__(self, data, batch_size, shuffle, validation_split, num_workers, collate_fn=default_collate,
                 extra_val_datasets=None, drop_last=False,
                 data_split=True, cross_validation_seed=0):
        assert data_split, "Handling when data is not split into cons/low/high data currently not implemented"
        self.validation_split = validation_split
        self.cross_validation_seed = cross_validation_seed
        self.extra_val = extra_val_datasets
        cons, low, high = data
        folds = 1/validation_split
        if not folds.is_integer(): print(f'Warning: rounded down to {folds} cross-validation folds')
        self.train, self.test_all, self.test_low, self.test_high, \
            self.val_all = self.get_train_test_and_val_sets(cons, low, high)

        self.init_kwargs = {
            'batch_size': batch_size,
            'shuffle': shuffle,
            'collate_fn': collate_fn,
            'num_workers': num_workers,
            'drop_last': drop_last
        }

        super().__init__(dataset=self.train, **self.init_kwargs)

    def split_validation(self):
        if not self.extra_val:
            # return three dataloaders here based on my validation datasets
            return DataLoader(dataset=self.test_all,  **self.init_kwargs),\
                   DataLoader(dataset=self.test_low, **self.init_kwargs),\
                   DataLoader(dataset=self.test_high,  **self.init_kwargs),\
                   DataLoader(dataset=self.val_all,  **self.init_kwargs)#,\
                   # DataLoader(dataset=self.test_low, **self.init_kwargs),\
                   # DataLoader(dataset=self.test_high,  **self.init_kwargs)
        else:
            # three validation sets which are standard in CV
            # todo: fix
            raise Exception('Not yet updated to test set ages')
            val_sets = [DataLoader(dataset=self.val_all,  **self.init_kwargs),
                   DataLoader(dataset=self.val_low, **self.init_kwargs),
                   DataLoader(dataset=self.val_high,  **self.init_kwargs)]
            # 6 other validation sets which I have added
            if len(self.extra_val) != 9: raise Exception('Unexpected number of extra validation sets')
            for extra_val_set in self.extra_val:
                val_sets.append(DataLoader(dataset=extra_val_set,  **self.init_kwargs))
            return val_sets

    def get_train_test_and_val_sets(self, cons, low, high, folds=10):
        cons_fold_len = int(cons.shape[0] / folds)
        high_fold_len = int(high.shape[0] / folds)
        low_fold_len = int(low.shape[0] / folds)

        # 1 fold for validation & early stopping
        fold_val = self.cross_validation_seed
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
        print(f'Size mixed validation dataset: {len(val_all)}')
        print(f'Size low inclusion validation dataset: {len(val_low)}')
        print(f'Size high inclusion validation dataset: {len(val_high)}')

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