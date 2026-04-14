from pytorch.utils.util import load_hs_tiff

import os
import numpy as np

DIR_LOAD = 'data/data_raw/hyperleaf'
DIR_SAVE = 'data/data_numpy/hyperleaf'

tiff_filenames = [f for f in os.listdir(DIR_LOAD) if not os.path.isdir(f)]

for tiff_filename in tiff_filenames:
    load_path = os.path.join(DIR_LOAD, tiff_filename)

    numpy_data = load_hs_tiff(load_path)

    save_path = os.path.join(DIR_SAVE, tiff_filename.split('.')[0] + '.npy')
    np.save(save_path, numpy_data)