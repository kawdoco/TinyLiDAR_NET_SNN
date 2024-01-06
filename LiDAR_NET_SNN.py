
import torch
import numpy as np
import torch.nn as nn
from torch.nn import Conv1d, BatchNorm2d,Conv2d,Softmax,BatchNorm1d, Linear, Conv2d, Flatten
from torch.nn import ReLU,Sigmoid,Softmax
from torch.nn import Module,Sequential
from torch.nn.init import xavier_normal_
import torch.nn.functional as F

from spikingjelly.activation_based import (
    neuron as snn,
    surrogate as sg,
    functional as sf,
    layer as sl,
)

class LiDAR_NET_SNN(Module):
    def __init__(self):
        super(LiDAR_NET_SNN, self).__init__()
        self.conv1 = Conv2d(1,5,5,1)
        self.BN1 = nn.BatchNorm2d(5)
        self.IF1 = snn.IFNode()
        
        self.conv2 = Conv2d(5,5,3,1)
        self.BN2 = nn.BatchNorm2d(5)
        self.IF2 = snn.IFNode()
        
        self.conv3 = Conv2d(5,5,3,2)
        self.BN3 = nn.BatchNorm2d(5)
        self.IF3 = snn.IFNode()
        
        self.flatten = Flatten()
        
        self.fc1 = Linear(405,100)
        self.IF4 = snn.IFNode()
        
        self.fc2 = Linear(100,10)
        self.IF5 = snn.IFNode()
        
        # self.softmax = Softmax(dim=1)
                                 
    # forward propagate input
    def forward(self, x):
        x = self.conv1(x)
        x = self.BN1(x)
        x = self.IF1(x)
        
        x = self.conv2(x)
        x = self.BN2(x)
        x = self.IF2(x)
        
        x = self.conv3(x)
        x = self.BN3(x)
        x = self.IF3(x)
        
        x = self.flatten(x)
        
        x = self.fc1(x)
        x = self.IF4(x)
        
        x = self.fc2(x)
        gesture_hat = self.IF5(x)
        
        # gesture_hat = self.softmax(gesture_hat)
                
        return gesture_hat
    
    
class LiDAR_NET_SNN_MLP(Module):
    def __init__(self):
        super(LiDAR_NET_SNN_MLP, self).__init__()
        self.flatten = Flatten()
        self.fc1 = Linear(25*25,200)
        self.DO1 = nn.Dropout(0.2)
        self.IF1 = snn.IFNode()
        
        self.fc2 = Linear(200,10)
        self.DO2 = nn.Dropout(0.2)
        self.IF2 = snn.IFNode()
                                 
    # forward propagate input
    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.DO1(x)
        x = self.IF1(x)
        
        x = self.fc2(x)
        x = self.DO2(x)
        gesture_hat = self.IF2(x)
                
        return gesture_hat
    