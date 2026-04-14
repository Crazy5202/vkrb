from pytorch.utils.util import load_pkl #load_hs_tiff, hs_to_tensor_clipping_scaling
from torch.utils.data import Dataset

import pandas as pd
import os
from torchvision.transforms import v2
import torch
import numpy as np

class HyperTiffDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file, dtype={"ImageId": str})
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, str(self.img_labels.iloc[idx, 0]) + ".npy")
        image = torch.from_numpy(np.load(img_path, mmap_mode = 'r+'))
        label = torch.Tensor(self.img_labels.iloc[idx, -4:].values.astype(int))
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

val_split = 0.2

preprocess_dict = load_pkl('preprocess')

if preprocess_dict['val_size'] != val_split:
    print(f"Pre-calculated validation split {preprocess_dict['val_size']} doesn't match current value {val_split}")
    raise RuntimeError

transform_func = v2.Compose([
    #v2.RandomResizedCrop(size=(224, 224), scale=(0.2, 1.0), ratio=(0.75, 1.333)),
    #v2.RandomHorizontalFlip(p=0.5),
    v2.ToDtype(torch.float32),
    v2.Normalize(mean=preprocess_dict['mean'], std=preprocess_dict['std']),
])

# train_dataset = HyperTiffDataset(annotations_file = "data/data_csv/hyperleaf/train.csv", img_dir = "data/data_raw/hyperleaf", transform = hs_to_tensor_clipping_scaling)
# train_dataloader = BaseDataLoader(train_dataset, batch_size=64, shuffle=False, validation_split=val_split)

# test_dataset = HyperTiffDataset(annotations_file = "data/data_csv/hyperleaf/solution.csv", img_dir = "data/data_raw/hyperleaf", transform = hs_to_tensor_clipping_scaling)
# test_dataloader = BaseDataLoader(test_dataset, batch_size=64, shuffle=True)

# raw_dataset = HyperTiffDataset(annotations_file = "data/data_csv/hyperleaf/solution.csv", img_dir = "data/data_raw/hyperleaf")
# raw_dataloader = BaseDataLoader(raw_dataset, batch_size=64, shuffle=True, validation_split=val_split)