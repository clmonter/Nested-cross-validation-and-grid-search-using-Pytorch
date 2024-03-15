
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 09:44:10 2024

@author: clmonter
"""

from torch import nn
import torch
import functions
from dataset_class import *
import torch

class SimpleNet(nn.Module):
    def __init__(self, dimx, hidden1, dimy):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(dimx, hidden1),
            nn.ReLU(),
            #nn.BatchNorm1d(hidden1),
            nn.Linear(hidden1, dimy),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        x = x.to(torch.float32)  # Convert to torch.float32
        y = self.layers(x)
        return y

class MultiHeadNet(nn.Module):
    def __init__(self, dimx, hidden1, dimy):
        super().__init__()

        self.attention_layer = nn.MultiheadAttention(embed_dim=dimx, num_heads=17)

        self.layers = nn.Sequential(
            nn.Linear(dimx, hidden1),
            nn.BatchNorm1d(hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, dimy),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        x = x.to(torch.float32)  # Convert to torch.float32
        x, _ = self.attention_layer(x, x, x)
        y = self.layers(x)
        return y
