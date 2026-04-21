import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data.dataloader import default_collate
from typing import Tuple, Optional


class BaseDataLoader(DataLoader):
    """
    Base class for all data loaders
    """
    def __init__(self, dataset, batch_size, shuffle = True, validation_split = 0.0, 
                collate_fn=default_collate, num_workers = 0, prefetch_factor=None, 
                pin_memory=False, persistent_workers=False, drop_last=False,
                val_dataset=None, val_batch_size = None, val_workers = None, val_prefetch = None,):
        self.validation_split = validation_split
        self.shuffle = shuffle

        self.batch_idx = 0
        self.n_samples = len(dataset)

        self.sampler, self.valid_sampler = self._split_sampler(self.validation_split)

        self.main_kwargs = {
            'dataset': dataset,
            'batch_size': batch_size,
            'shuffle': self.shuffle,
            'collate_fn': collate_fn,
            'num_workers': num_workers,
            'pin_memory': pin_memory,
            'prefetch_factor': prefetch_factor,
            'persistent_workers': persistent_workers,
            'drop_last': drop_last
        }

        if self.valid_sampler is not None:
            self.val_kwargs = {
                'dataset': val_dataset if val_dataset is not None else dataset,
                'batch_size': val_batch_size if val_batch_size is not None else batch_size,
                'shuffle': self.shuffle,
                'collate_fn': collate_fn,
                'num_workers': val_workers if val_workers is not None else num_workers,
                'pin_memory': pin_memory,
                'prefetch_factor': val_prefetch if val_prefetch is not None else prefetch_factor,
                'persistent_workers': persistent_workers,
                'drop_last': drop_last
            }

        super().__init__(sampler=self.sampler, **self.main_kwargs)

    def _split_sampler(self, split) -> Tuple[Optional[SubsetRandomSampler], Optional[SubsetRandomSampler]]:
        if split == 0.0:
            return None, None

        idx_full = np.arange(self.n_samples)

        rng = np.random.default_rng(0)
        rng.shuffle(idx_full)
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

        self.shuffle = False
        self.n_samples = len(train_idx)

        return train_sampler, valid_sampler

    def split_validation(self) -> Optional[DataLoader]:
        if self.valid_sampler is None:
            return None
        else:
            return DataLoader(sampler=self.valid_sampler, **self.val_kwargs)
