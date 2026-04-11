import pytorch.imp.dataloader_raw as raw_data_module
from pytorch.utils.util import save_pkl, load_pkl

import torch

# mean = []
# std = []

#print(raw_data_module.val_split)

# for _, (data, _) in enumerate(raw_data_module.raw_dataloader):
#     for elem in data:
#         print(elem.shape)
#         count += 1
# print(count)

# for batch in raw_data_module.raw_dataloader:
#     for i in range(len(batch)):
#         print(batch[i][0].shape)

mean = torch.zeros(raw_data_module.raw_dataloader.dataset[0][0].shape[0], dtype=torch.float64)
std  = torch.zeros(raw_data_module.raw_dataloader.dataset[0][0].shape[0], dtype=torch.float64)

total_pixels = 0

for i, (data, _) in enumerate(raw_data_module.raw_dataloader):
    b, c, h, w = data.shape
    nb_pixels = b * h * w

    mean += data.sum(dim=[0, 2, 3]).double()
    std  += (data ** 2).sum(dim=[0, 2, 3]).double()

    total_pixels += nb_pixels

mean /= total_pixels
mean_of_sq = std / total_pixels

std = (mean_of_sq - mean ** 2).sqrt()

save_dict = {'val_size': raw_data_module.val_split, 'mean': mean.tolist(), 'std': std.tolist()}

print(save_dict)

save_pkl(data=save_dict, name='preprocess')

# try_dict = load_pkl(name='preprocess')
# print(try_dict)