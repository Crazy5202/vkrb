from pytorch.utils.util import load_pkl #load_hs_tiff, hs_to_tensor_clipping_scaling

from torch.utils.data import Dataset
import pandas as pd
import os
from torchvision.transforms import v2
import torch
import numpy as np

class OriginalDataset(Dataset):
    def __init__(self, annotations_file, img_dir):
        self.img_labels = pd.read_csv(annotations_file, dtype={"ImageId": str})
        self.img_dir = img_dir

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, str(self.img_labels.iloc[idx, 0]) + ".npy")
        image = torch.from_numpy(np.load(img_path, mmap_mode = 'r+'))
        label = torch.Tensor(self.img_labels.iloc[idx, -4:].values.astype(int))
        return image, label
    
class TransformDataset(Dataset):
    def __init__(self, base_dataset: Dataset, transform = None, target_transform = None):
        self.base_dataset = base_dataset
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        image, label = self.base_dataset[idx]
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

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

transform_train = v2.Compose([
    v2.ToDtype(torch.float32),
    v2.RandomHorizontalFlip(p=0.5),
    v2.RandomVerticalFlip(p=0.5),
    v2.RandomAffine(
        degrees=0, 
        translate=(0.25, 0.25), 
        scale=(0.75, 1.25),
        fill=0
    ),
    v2.GaussianBlur(kernel_size=5),
    AddGaussianNoise(mean=0., std=0.01),
    v2.Normalize(mean=preprocess_dict['mean'], std=preprocess_dict['std']),
])

transform_val = v2.Compose([
    v2.ToDtype(torch.float32),
    v2.Normalize(mean=preprocess_dict['mean'], std=preprocess_dict['std']),
])

train_val_raw_dataset = OriginalDataset(annotations_file = "data/data_csv/hyperleaf/train.csv", img_dir = "data/data_numpy/hyperleaf")
train_dataset = TransformDataset(train_val_raw_dataset, transform_train)
val_dataset = TransformDataset(train_val_raw_dataset, transform_val)

test_raw_dataset = OriginalDataset(annotations_file = "data/data_csv/hyperleaf/solution.csv", img_dir = "data/data_numpy/hyperleaf")
test_dataset = TransformDataset(test_raw_dataset, transform_val)