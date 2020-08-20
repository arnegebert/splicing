from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate


class BaseDataLoader(DataLoader):
    """
    Base class for all data loaders
    """
    def __init__(self, dataset, batch_size, shuffle, validation_split, num_workers, collate_fn=default_collate,
                 extra_val_datasets=None, drop_last=False):
        self.validation_split = validation_split
        self.extra_val = extra_val_datasets
        self.train, self.test_all, self.test_low, self.test_high, self.val_all = dataset

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

    def get_train_test_and_val_sets(self):
        pass