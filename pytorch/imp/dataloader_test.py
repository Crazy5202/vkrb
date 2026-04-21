from pytorch.src.base_dataloader import BaseDataLoader
from pytorch.imp.data import test_dataset

test_dataloader = BaseDataLoader(dataset=test_dataset, batch_size=16, num_workers = 2, prefetch_factor = 2, 
                                 persistent_workers=True, pin_memory = True)
# val_dataloader = BaseDataLoader(dataset=val_dataset, batch_size=32, num_workers = 2, prefetch_factor = 2, 
#                                  persistent_workers=True, pin_memory = True)