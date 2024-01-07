
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
        conv1_spikes = torch.count_nonzero(x[1,:,:,:]) #(151*8)/(25*25*5)=0.386, 0.386x1x21x21x5x5x5=0.386x55125=21278
        x = self.conv1(x) 
        x = self.BN1(x)
        x = self.IF1(x)
        
        conv2_spikes = torch.count_nonzero(x[1,:,:,:]) #(453*8)/(21*21*5) = 1.643, 1.643x5x19x19x5x5x5=1.643x225625=370702
        x = self.conv2(x)
        x = self.BN2(x)
        x = self.IF2(x)
        
        conv3_spikes = torch.count_nonzero(x[1,:,:,:]) #(445*8)/(19*19*5) = 1.972, 1.9725x9x9x5x5x5=1.972x50625=99832
        x = self.conv3(x)
        x = self.BN3(x)
        x = self.IF3(x)
        
        x = self.flatten(x)
        
        fc1_spikes = torch.count_nonzero(x[1,:]) #(122*8)/(405) = 2.410, 2.410x405x100=97605
        x = self.fc1(x)
        x = self.IF4(x)
        
        fc2_spikes = torch.count_nonzero(x[1,:]) #(41*8)/(100) = 3.20, 3.20x100x10=3200
        x = self.fc2(x)
        gesture_hat = self.IF5(x)
        
        # gesture_hat = self.softmax(gesture_hat)
                
        return gesture_hat,conv1_spikes,conv2_spikes,conv3_spikes,fc1_spikes,fc2_spikes
    
    
class LiDAR_NET_SNN_MLP(Module):
    def __init__(self):
        super(LiDAR_NET_SNN_MLP, self).__init__()
        self.flatten = Flatten()
        self.fc1 = Linear(25*25,200) # FLOPs = 625 x 200 x 2 = 250000, 250000x1.894=
        self.DO1 = nn.Dropout(0.2)
        self.IF1 = snn.IFNode()
        
        self.fc2 = Linear(200,10) # FLOPs = 200 x 10 x 2 = 4000, 4000x2.28=
        self.DO2 = nn.Dropout(0.2)
        self.IF2 = snn.IFNode()
                                     #total FLOPs = 254000
    # forward propagate input
    def forward(self, x):
        x = self.flatten(x)
        fc1_spikes=torch.count_nonzero(x[1,:]) #148*8/625=1.894
        x = self.fc1(x)
        x = self.DO1(x)
        x = self.IF1(x)
        
        fc2_spikes=torch.count_nonzero(x[1,:]) #57*8/200 = 2.28
        x = self.fc2(x)
        x = self.DO2(x)
        gesture_hat = self.IF2(x)
                
        return gesture_hat
    