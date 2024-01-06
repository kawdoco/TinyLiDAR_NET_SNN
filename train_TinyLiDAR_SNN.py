#!/usr/bin/env python3
# -*- coding: utf-8 -*-"
import time,os
import torch
import numpy as np
import scipy.io as io
from scipy.io import savemat
import datetime
import matplotlib.pyplot as plt
import math

from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torch.optim import RMSprop,SGD, Adam
from torch.nn import MSELoss, CrossEntropyLoss
import torch.nn.functional as F
from EarlyStopping import EarlyStopping

from LiDAR_NET_CNN import LiDAR_NET_CNN
from LiDAR_NET_SNN import LiDAR_NET_SNN,LiDAR_NET_SNN_MLP

import torch.optim.lr_scheduler as lr_scheduler

from torchvision import datasets, transforms
import torchvision.transforms as transforms
from utils import poisson_encode

from spikingjelly.activation_based import (
    neuron as snn,
    surrogate as sg,
    functional as sf,
    layer as sl,
)

#%%
#-----------------------------------------------------------------------------#
def load_data(path, data_name, label_name, test_ratio,BATCH_SIZE):
    DataSet = np.load(os.path.join(path,data_name))
    indata = np.transpose(DataSet, (2,0,1))
    indata = np.expand_dims(indata, 1)
    # indata = indata.reshape((5000, 1, -1))

    label = np.load(os.path.join(path,label_name))
    label = np.transpose(label, (1,0))

    print('Input train-data size',indata.shape)
    print('Input train-data-label size',label.shape)
    
    indata = torch.from_numpy(indata)
    targets = torch.from_numpy(label)
    targets= targets.type(torch.FloatTensor) 
    torch_dataset = TensorDataset(indata,targets)
    test_size = round(test_ratio * len(indata))
    train_size =  len(indata) - test_size
    train, test = random_split(torch_dataset, [train_size,test_size])

    train_set = DataLoader(dataset=train,batch_size=BATCH_SIZE,shuffle=True,num_workers=0)
    test_set = DataLoader(dataset=test,batch_size=BATCH_SIZE,shuffle=True,num_workers=0)
    
    return train_set,test_set

def train_model(train_set, model,Epoch,USE_GPU, log_interval, patience, lr, T):

    # criterion = MSELoss().cuda()
    criterion = CrossEntropyLoss().cuda()
    optimizer = Adam(model.parameters(), lr=lr)
    # scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    
    # Record the loss 
    Train_Loss_Total = list()
    Val_Loss_Total = list()
    
    Train_Acc_Total = list()
    Val_Acc_Total = list()
    
    # Enumerate epochs
    start_time = time.time()
    early_stopping = EarlyStopping(patience=patience, verbose=True)
    #Epoch loop
    for epoch in range(Epoch):
        print('Epoch {}/{}'.format(epoch+1, Epoch))
        
        # adjust_learning_rate(optimizer, epoch)
        for param_group in optimizer.param_groups:
            print('lr: ',param_group['lr'])
        
        #Do training, then validating
        for phase in ['train', 'val']:
            tic = time.time()
            
            acc_train = list()
            acc_val = list()
            
            if phase == 'train':
                model.train()  # Set model to training mode
                use_set = train_set
            else:
                model.eval()   # Set model to evaluating mode
                use_set = test_set
            # enumerate mini batches
            for batch_idx, (inputs, targets) in enumerate(use_set):
                #running loss
                yhat = torch.zeros(targets.size(0), targets.size(1)).cuda() if USE_GPU else torch.zeros(targets.size(0), targets.size(1))
                l = list()
                targets=targets.type(torch.LongTensor)
                if USE_GPU:
                    inputs, targets = inputs.cuda(), targets.cuda()
                # clear the gradients
                optimizer.zero_grad()
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    inputs_T = poisson_encode(inputs,T)
                    sf.reset_net(model)
                    for xt in inputs_T:
                        yhat += model(xt.float())
                        
                    yhat = yhat.softmax(-1)
                    loss = criterion(yhat.float(), targets.float())
                    _, predicted = torch.max(yhat, 1)
                    _, gt = torch.max(targets,1)
                    correct = (predicted == gt).sum().item()
                    total = targets.size(0)
                    accuracy = correct / total
                    acc_train.append(accuracy)
                    
                if  phase == 'train':        
                    # credit assignment
                    loss.backward()
                    # update model weights
                    optimizer.step()
                    
                #Convert tensor to numpy
                l.append(loss.cpu().detach().numpy())
                
                if (phase == 'val'):
                    acc_val.append(accuracy)
                
                if (batch_idx % log_interval == 0 and phase == 'val'):
                    print('Validation accuracy: {:.2f}%'.format(accuracy*100))
                
                if (batch_idx % log_interval == 0 and phase == 'train'):
                    print('Train Epoch: {} [{}/{} ({:.1f}%)]\tLoss: {:.6f}\tAccuracy: {:.2f}%'.format(
                        epoch+1, batch_idx * len(inputs), len(use_set.dataset), 
                        100. * batch_idx / len(use_set), loss, accuracy*100))
            
            if phase == 'train':
                Train_Loss_Total.append(np.mean(l))
                Train_Acc_Total.append(np.mean(acc_train))
            else:
                Val_Loss_Total.append(np.mean(l))
                Val_Acc_Total.append(np.mean(acc_val))
                
                early_stopping(epoch+1, np.mean(l), model, ckpt_dir) 
             
            toc = time.time()-tic
            
            # scheduler.step()
            
            print('\n{} Loss: {:.6f} time: {:.4f}s'.format(phase, np.mean(l), toc))
            if phase == 'val':        
                print('-' * 50)
                
        if early_stopping.early_stop: 
            print("Early stopping")
            break    
        Record = {'Train_Loss_Total':Train_Loss_Total,'Val_Loss_Total':Val_Loss_Total}
    
    stop_time = time.time() - start_time
    
    np.savez('Loss_SNN_CNN', \
             Train_Loss_Total=Train_Loss_Total,Val_Loss_Total=Val_Loss_Total, \
             Train_Acc_Total=Train_Acc_Total,Val_Acc_Total=Val_Acc_Total)
    
    print("Total training time: {:.2f}".format(stop_time))
    
    return [Record,Train_Loss_Total,Val_Loss_Total,Train_Acc_Total,Val_Acc_Total,epoch]
         
#%%   
#-----------------------------------------------------------------------------#
if __name__ == '__main__':  
    path = r'./'   
    data_name = r'datasets_total.npy'
    label_name = r'label.npy'
    
    train_set, test_set = load_data(path, data_name,label_name,test_ratio = 0.2,BATCH_SIZE = 32)  
    
    USE_GPU = False

    ckpt_dir='./LiDAR_NET_Pre_trained_model_SNN_CNN'
    os.makedirs(ckpt_dir,exist_ok=True)
    model = LiDAR_NET_SNN() #LiDAR_NET_SNN LiDAR_NET_SNN_MLP
        
    if USE_GPU:
        model.cuda() 
    #paramter_initialize(model)
    [Record, Train_Loss_Total,Val_Loss_Total,Train_Acc_Total,Val_Acc_Total,epoch]= \
    train_model(train_set, model,Epoch=300, USE_GPU = USE_GPU,log_interval =50,patience=20,lr=0.001, T=8)    
    
     
