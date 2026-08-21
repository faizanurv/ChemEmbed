# data_loaders/spectra_inference_dataset_loader.py

import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

class class_ls(Dataset):
    def __init__(self, x, transform=None):
        df = pd.read_pickle(x)
        self.X = df['coo_format_data'].tolist()
        self.N = df['Precursor'].tolist()
        self.ik = df['Unique_ID'].tolist()

    def __getitem__(self, index):
        feature = self.X[index]
        t1 = feature.todense()
        t1 = t1.astype(np.float32)
        t1 = torch.from_numpy(t1)
        non_zero_indices = torch.nonzero(t1)
        lenght_spectra = non_zero_indices.shape[0]
        
        unique_id = self.ik[index]

        return t1, self.N[index], lenght_spectra, unique_id

    def __len__(self):
        return len(self.X)
