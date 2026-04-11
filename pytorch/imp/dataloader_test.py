from pytorch.src.base_dataloader import BaseDataLoader
from pytorch.imp.data import HyperTiffDataset, transform_func
#from pytorch.utils.util import hs_to_tensor_clipping_scaling

test_dataset = HyperTiffDataset(annotations_file = "data/data_csv/hyperleaf/solution.csv", img_dir = "data/data_raw/hyperleaf", transform = transform_func)
test_dataloader = BaseDataLoader(test_dataset, batch_size=64, shuffle=True)