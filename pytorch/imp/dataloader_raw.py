from pytorch.src.base_dataloader import BaseDataLoader
from pytorch.imp.data import train_val_raw_dataset, val_split

# def no_collate(batch):
#     return batch

raw_dataloader = BaseDataLoader(dataset=train_val_raw_dataset, batch_size=64, 
                                validation_split=val_split) #, collate_fn=no_collate)