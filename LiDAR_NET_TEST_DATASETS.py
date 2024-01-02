# -*- coding: utf-8 -*-
"""
Created on Sat Sep 16 15:35:36 2023

@author: Shufan_Young
"""

import os
import torch
import numpy as np
import scipy.io as io
from LiDAR_NET_CNN import LiDAR_NET_CNN
from torch.autograd import Variable
import matplotlib.pyplot as plt
from scipy.io import savemat
import time
import seaborn as sns
import pandas as pd
from sklearn import metrics
#%%
torch.set_printoptions(precision=10)

no_samples=1000
no_gestures=10

frame=np.load('datasets_test.npy')  
gesture_gt=np.load('label_test.npy')   
gesture_gt=np.transpose(gesture_gt,(1,0))

frame=frame.astype(np.float32)

frame = Variable(torch.from_numpy(frame))
frame = frame.unsqueeze(0)
frame=np.transpose(frame, (3,0,1,2))

#use gpu
PATH=r'./LiDAR_NET_Pre_trained_model/ckpt_epoch_68_val_loss_1.461150.pth'
model = LiDAR_NET_CNN()

#use cpu
#checkpoint = torch.load(PATH, map_location='cpu')
#model.load_state_dict(checkpoint)

checkpoint=torch.load(PATH)
model.load_state_dict(checkpoint,strict=False)
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
frame_torch = frame.to(device)

gesture_gt_hat=torch.zeros(no_samples,1)

start = time.time()
with torch.no_grad():
    # frame_torch_element = frame_torch_element.unsqueeze(0)
    gesture_gt_hat = model(frame_torch)
end = time.time()

max_index=np.zeros(no_samples)
for i in range(no_samples):
    max_value = gesture_gt_hat[i,0] 
    for j in range(no_gestures):
        if gesture_gt_hat[i,j] > max_value:
            max_value = gesture_gt_hat[i,j]
            max_index[i] = j

max_index_gt = np.zeros(no_samples)
for i in range(no_samples):
    max_index_gt[i] = np.argmax(gesture_gt[i,:])

print('------Inference Done------')
print(f"Runtime of the program is {end - start}")# -*- coding: utf-8 -*-
#%% Accuracy evaluation
from sklearn.metrics import accuracy_score, confusion_matrix

accuracy = accuracy_score(max_index_gt, max_index)
cm = confusion_matrix(max_index_gt, max_index, labels=np.arange(0,10))

plt.figure(figsize=(8, 6))
sns.heatmap(cm, fmt="d", cmap="Blues", cbar=True,vmin=0, vmax=100)
plt.xticks(ticks=np.arange(0,10), labels=np.arange(0,10)+1)
plt.yticks(ticks=np.arange(0,10), labels=np.arange(0,10)+1)
plt.title(f"CNN Confusion Matrix\nAccuracy: 82.7%")
plt.xlabel("Predicted gestures")
plt.ylabel("True gestures")
plt.tight_layout()
dpi_value = 300  # Adjust this value as needed
plt.savefig(r'./Figures/CNN_confusion_matrix.png', dpi=dpi_value)
plt.show()