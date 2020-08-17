import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import SubsetRandomSampler


class BaseDataLoader(DataLoader):
    """
    Base class for all data loaders
    """
    def __init__(self, datasets, batch_size, shuffle, validation_split, num_workers, dsc_cv=False, collate_fn=default_collate,
                 extra_val_datasets=None):
        self.validation_split = validation_split
        self.shuffle = shuffle

        self.batch_idx = 0
        self.n_samples = len(datasets)
        self.dsc_cv = dsc_cv
        self.dataset = datasets
        self.extra_val = extra_val_datasets
        self.sampler, self.valid_sampler = self._split_sampler(self.validation_split)

        if dsc_cv:
            self.shuffle = shuffle

        self.init_kwargs = {
            'batch_size': batch_size,
            'shuffle': self.shuffle,
            'collate_fn': collate_fn,
            'num_workers': num_workers,
            'drop_last': False
        }

        if not dsc_cv:
            super().__init__(sampler=self.sampler, dataset=datasets, **self.init_kwargs)
        else:
            # choose a pytorch thing that suits me
            # use self.samples here
            self.train, self.val_sets, self.test_sets = datasets
            super().__init__(dataset=self.train, **self.init_kwargs)

    def _split_sampler(self, split):
        if split == 0.0:
            return None, None

        idx_full = np.arange(self.n_samples)

        np.random.seed(0)
        np.random.shuffle(idx_full)

        if isinstance(split, int):
            assert split > 0
            assert split < self.n_samples, "validation set size is configured to be larger than entire dataset."
            len_valid = split
        else:
            len_valid = int(self.n_samples * split)

        valid_idx = idx_full[0:len_valid]
        train_idx = np.delete(idx_full, np.arange(0, len_valid))

        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        # turn off shuffle option which is mutually exclusive with sampler
        self.shuffle = False
        self.n_samples = len(train_idx)

        return train_sampler, valid_sampler

    def split_validation(self):
        if self.valid_sampler is None:
            return None
        else:
            if not self.dsc_cv:
                raise Exception('Deprecated functionality')
                return DataLoader(sampler=self.valid_sampler, dataset=self.dataset, **self.init_kwargs)
            elif self.dsc_cv and not self.extra_val:
                # return three dataloaders here based on my validation datasets
                val_loaders = [DataLoader(dataset=dataset,  **self.init_kwargs) for dataset in self.val_sets]
                test_loaders = [DataLoader(dataset=dataset,  **self.init_kwargs) for dataset in self.test_sets]
                return val_loaders, test_loaders

            else:
                # three validation sets which are standard in CV
                val_loaders = [DataLoader(dataset=dataset,  **self.init_kwargs) for dataset in self.val_sets]
                test_loaders = [DataLoader(dataset=dataset,  **self.init_kwargs) for dataset in self.test_sets]
                # 6 other validation sets which I have added
                if len(self.extra_val) != 9: raise Exception('Unexpected number of extra validation sets')
                for extra_val_set in self.extra_val:
                    val_loaders.append(DataLoader(dataset=extra_val_set,  **self.init_kwargs))
                return val_loaders, test_loaders
