from pytorch.src.base_dataloader import BaseDataLoader
from pytorch.imp.data import HyperTiffDataset, val_split

# def no_collate(batch):
#     return batch

raw_dataset = HyperTiffDataset(annotations_file = "data/data_csv/hyperleaf/train.csv", img_dir = "data/data_numpy/hyperleaf")
raw_dataloader = BaseDataLoader(raw_dataset, batch_size=64, validation_split=val_split) #, collate_fn=no_collate)