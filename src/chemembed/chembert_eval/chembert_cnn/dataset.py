import torch
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
import os
from torchvision import transforms
import pandas as pd
import rdkit
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem.Descriptors import MolWt, ExactMolWt
import decimal
from numpy.linalg import norm
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d,Dropout
from torch.optim import Adam, SGD
import pickle
from torch.utils.data import DataLoader
from rdkit.Chem import rdMolDescriptors

from torch.utils.data import Dataset

class class_ls(Dataset):

    def __init__(self, x, transform=None):
        
        df = x

        #df = df[df['Tanimato'] == 1]
        self.X = df['coo_format_data'].tolist()
        #self.X = df['spectra_embedding'].tolist()

        self.N = df['Precursor'].tolist()

        self.sm = df['smile'].tolist()

        #self.Y = df['mol2vec'].tolist()
        #self.Y = df['DB'].tolist()

        self.ik = 0#df['Inchikey'].tolist()



    def __getitem__(self, index):
        # X
        feature= self.X[index]

        t1 = feature.todense()
        t1 = t1.astype(np.float32)
        t1 = torch.from_numpy(t1)

        # Get indices of non-zero elements
        non_zero_indices = torch.nonzero(t1)

        # Extract non-zero values
        lenght_spectra = non_zero_indices.shape[0]

        #  smile

        smile = self.sm[index]

        label = 0

        return t1,label,smile,lenght_spectra,0#self.ik[index]

    def __len__(self):
        return (len(self.X))





