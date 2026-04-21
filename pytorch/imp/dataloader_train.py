from pytorch.src.base_dataloader import BaseDataLoader
from pytorch.imp.data import val_split, train_dataset, val_dataset

train_dataloader = BaseDataLoader(dataset=train_dataset, batch_size=16,
                                num_workers = 6, prefetch_factor = 2,
                                persistent_workers=True, pin_memory = True,
                                validation_split=val_split, val_dataset=val_dataset, val_workers=2)