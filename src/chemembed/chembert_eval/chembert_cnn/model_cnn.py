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

import torch.nn as nn
import torch.nn.functional as F

class CNN_Class(nn.Module):
    def __init__(self, featuredim=420, output=28):
        super(CNN_Class, self).__init__()
        self.conv1 = nn.Conv1d(1, 64, 3, 1, 1)
        self.pool1 = nn.MaxPool1d(2, 2)
        self.conv2 = nn.Conv1d(64, 128, 3, 1, 1)
        self.pool2 = nn.MaxPool1d(2, 2)
        self.conv3 = nn.Conv1d(128, 128, 3, 1, 1)
        self.pool3 = nn.MaxPool1d(2, 2)

        self.conv4 = nn.Conv1d(128, 128, 3, 1, 1)
        self.pool4 = nn.MaxPool1d(2, 2)

        self.conv5 = nn.Conv1d(128, 128, 3, 1, 1)
        self.pool5 = nn.MaxPool1d(2, 2)

        self.conv6 = nn.Conv1d(128, 128, 3, 1, 1)
        self.pool6 = nn.MaxPool1d(2, 2)



        self.fc1 = nn.Linear(139904, 2000)  # 139904 equal spacing between in/out variables for FC layers

        self.fc2 = nn.Linear(2000, 1024)

        self.fc3 = nn.Linear(1024, 384)

       
        self.dropout = nn.Dropout(0.25)
       

    def encode(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv3(x)
        x = self.pool3(x)

        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv4(x)
        x = self.pool4(x)

        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv5(x)
        x = self.pool5(x)

        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv6(x)
        x = self.pool6(x)

        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)

        # For Precursor-mz
        #xx= self.fc_mz(pre_mz)
        #xx= F.relu(xx)

        #value_to_concat = torch.tensor([pre_mz], dtype=torch.float32,device=x.device)


        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout(x)
        #x = x + xx
        x = self.fc3(x)

        return x

    def forward(self, x):
        x = self.encode(x)
        return x

