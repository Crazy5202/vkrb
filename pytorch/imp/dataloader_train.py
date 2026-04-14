from pytorch.src.base_dataloader import BaseDataLoader
from pytorch.imp.data import HyperTiffDataset, val_split, transform_func
#from pytorch.utils.util import hs_to_tensor_clipping_scaling

train_dataset = HyperTiffDataset(annotations_file = "data/data_csv/hyperleaf/train.csv", 
                                 img_dir = "data/data_numpy/hyperleaf", transform = transform_func)
train_dataloader = BaseDataLoader(train_dataset, batch_size=8, num_workers = 4, prefetch_factor = 2, 
                                  persistent_workers=True, pin_memory = True,
                                  validation_split=val_split, )