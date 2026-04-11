import json
import torch
import pandas as pd
from pathlib import Path
import numpy as np
import tifffile
import logging
import logging.config
import pickle
from typing import Any

# def ensure_dir(dirname):
#     dirname = Path(dirname)
#     if not dirname.is_dir():
#         dirname.mkdir(parents=True, exist_ok=False)

def read_json(fname) -> dict:
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle)

def write_json(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)

def setup_logging(save_dir, log_config='./logger_config.json', default_level=logging.INFO):
    """
    Setup logging configuration
    """
    log_config = Path(log_config)
    if log_config.is_file():
        config = read_json(log_config)
        for _, handler in config['handlers'].items():
            if 'filename' in handler:
                handler['filename'] = str(save_dir / handler['filename'])

        logging.config.dictConfig(config)
    else:
        print("Warning: logging configuration file is not found in {}.".format(log_config))
        logging.basicConfig(level=default_level)

def load_hs_tiff(file_path: str) -> np.ndarray:
    """
    Загрузка файла.
    """ ###
    tif = tifffile.TiffFile(file_path)
    spectral_stack = np.stack([page.asarray() for page in tif.pages], axis=0).astype(np.float32)
    return spectral_stack

def save_pkl(data, name: str, save_dir: str = ".") -> None:
    save_path = str(Path(save_dir) / (name + '.pkl'))
    with open(save_path, 'wb') as f:
        pickle.dump(data, f)

def load_pkl(name: str, load_dir: str = ".") -> Any:
    load_path = str(Path(load_dir) / (name + '.pkl'))
    with open(load_path, 'rb') as f:
        data = pickle.load(f)
    return data

def hs_to_tensor_clipping_scaling(img: np.ndarray) -> torch.Tensor:   
    eps = 1e-9

    q01 = np.percentile(img, 1)
    q99 = np.percentile(img, 99)
    
    img_clipped = np.clip(img, q01, q99)
    
    img_normalized = (img_clipped - q01) / (q99 - q01 + eps)
    
    tensor = torch.from_numpy(img_normalized.transpose((2, 0, 1))).contiguous().float()

    return tensor

def prepare_device(n_gpu_use):
    """
    setup GPU device if available. get gpu device indices which are used for DataParallel
    """
    n_gpu = torch.cuda.device_count()
    if n_gpu_use > 0 and n_gpu == 0:
        print("Warning: There\'s no GPU available on this machine,"
              "training will be performed on CPU.")
        n_gpu_use = 0
    if n_gpu_use > n_gpu:
        print(f"Warning: The number of GPU\'s configured to use is {n_gpu_use}, but only {n_gpu} are "
              "available on this machine.")
        n_gpu_use = n_gpu
    device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
    list_ids = list(range(n_gpu_use))
    return device, list_ids

class MetricTracker:
    def __init__(self, *keys, writer=None):
        self.writer = writer
        self._data = pd.DataFrame(index=keys, columns=['total', 'counts', 'average'])
        self.reset()

    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        if self.writer is not None:
            self.writer.add_scalar(key, value)
        self._data.loc[key, 'total'] += value * n
        self._data.loc[key, 'counts'] += n
        self._data.loc[key, 'average'] = self._data.total[key] / self._data.counts[key]

    def avg(self, key):
        return self._data.average[key]

    def result(self):
        return dict(self._data.average)