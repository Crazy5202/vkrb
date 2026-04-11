from pytorch.src.base_dataloader import BaseDataLoader
from pytorch.imp.data import HyperTiffDataset, val_split, transform_func
#from pytorch.utils.util import hs_to_tensor_clipping_scaling

train_dataset = HyperTiffDataset(annotations_file = "data/data_csv/hyperleaf/train.csv", img_dir = "data/data_raw/hyperleaf", transform = transform_func)
train_dataloader = BaseDataLoader(train_dataset, batch_size=64, validation_split=val_split)