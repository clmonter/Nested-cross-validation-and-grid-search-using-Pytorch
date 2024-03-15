#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 09:44:10 2024

@author: clmonter
"""

import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import torch

class CreateDataset(Dataset):
    def __init__(self, y, x):

        self.y = y #.astype(int)
        self.x = x

    def __len__(self): # length of dataset
        return len(self.y)

    def __getitem__(self, idx): 

        if torch.is_tensor(idx):
            idx = idx.tolist()

        #############################################################
        ## Labels
        y = self.y[idx]

        #############################################################
        ## Data
        x = self.x[idx,:]

        #############################################################
        
        sample = {'label': y.astype(np.int64), 'data': x} # Text

        return sample
