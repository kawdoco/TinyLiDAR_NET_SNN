# -*- coding: utf-8 -*-
"""
Created on Wed Sep  9 17:49:23 2020

@author: pc
"""

import torch
import numpy as np
import torch.nn as nn
from torch.nn import Conv1d, BatchNorm2d,Conv2d,Softmax,BatchNorm1d, Linear, Conv2d, Flatten
from torch.nn import ReLU,Sigmoid,Softmax
from torch.nn import Module,Sequential
from torch.nn.init import xavier_normal_
import torch.nn.functional as F

def Conv_Add(in_planes, out_planes, kernel_size, stride, padding=0, bias=False):
    return Conv2d(in_planes, out_planes, kernel_size,stride, padding=padding, bias=False)

    
class LiDAR_NET_CNN(Module):
    def __init__(self):
        super(LiDAR_NET_CNN, self).__init__()
        self.conv1 = Conv2d(1,5,5,1)#FLOPs=2x(1x21x21x5x5x5) = 110250
        self.bn1 = BatchNorm2d(5)
        
        self.conv2 = Conv2d(5,5,3,1)#FLOPs=2x(5x19x19x5x5x5) = 451250
        self.bn2 = BatchNorm2d(5)
        
        self.conv3 = Conv2d(5,5,3,2)#FLOPs=2x(5x9x9x5x5x5) = 101250
        self.bn3 = BatchNorm2d(5)
        
        self.flatten = Flatten()
        
        self.fc1 = Linear(405,100)#FLOPs=2x(405x100) = 81000
        self.fc2= Linear(100,10)#FLOPs=2x(100x10) = 2000
        
        self.softmax = Softmax(dim=1)
                                 # total flops=
    # forward propagate input
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        
        x = self.flatten(x)
        
        x = self.fc1(x)
        x = F.relu(x)
        gesture_hat = self.fc2(x)
        gesture_hat = self.softmax(gesture_hat)
                
        return gesture_hat
    
    
class LiDAR_NET_CNN_seq(nn.Module):
    def __init__(self):
        super(LiDAR_NET_CNN_seq, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 5, 5, 1),
            nn.BatchNorm2d(5),
            nn.ReLU(),

            nn.Conv2d(5, 5, 3, 1),
            nn.BatchNorm2d(5),
            nn.ReLU(),

            nn.Conv2d(5, 5, 3, 2),
            nn.BatchNorm2d(5),
            nn.ReLU(),

            nn.Flatten(),

            nn.Linear(405, 100),
            nn.ReLU(),

            nn.Linear(100, 10),
            nn.Softmax(dim=1)
        )

    # forward propagate input
    def forward(self, x):
        gesture = self.cnn(x)
        return gesture 