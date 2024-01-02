# -*- coding: utf-8 -*-
#ZZY 02/June/2021

import os
import torch
import numpy as np
import scipy.io as io
from LiDAR_NET_MLP import LiDAR_NET_MLP
from LiDAR_NET_CNN import LiDAR_NET_CNN
from torch.autograd import Variable
import matplotlib.pyplot as plt
from scipy.io import savemat
import time

np.set_printoptions(suppress=True)
torch.set_printoptions(precision=10)

DataSet = io.loadmat('one_frame.mat')
frame=DataSet.get('zero')    

frame=frame.astype(np.float32)

frame = Variable(torch.from_numpy(frame))
frame = frame.unsqueeze(0)
frame = frame.unsqueeze(1)

start = time.time()

PATH=r'LiDAR_NET_Pre_trained_model_CNN/ckpt_epoch_87_val_loss_0.037791.pth'
model = LiDAR_NET_CNN()

checkpoint=torch.load(PATH)
model.load_state_dict(checkpoint,strict=False)
model.eval()

#%%
with torch.no_grad():
    yhat = model(frame)
yhat=yhat.numpy()
predicted_gesture=np.argmax(yhat)
print(yhat)
print("Predicted gesture is", predicted_gesture)