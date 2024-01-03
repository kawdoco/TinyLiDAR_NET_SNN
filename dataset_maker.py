#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 31 17:36:18 2023

@author: apple
"""

import numpy as np
#%% wab training gen
img_size = 25
num_gestures = 10
data_size_per_g = 500
data_size = 5000

data_tensor = np.zeros((img_size,img_size,0))
data_nor = np.zeros((img_size,img_size,data_size_per_g))

for i in range(1,num_gestures+1):
    file_path = f"./datasets/{i}.npy"  # Using f-string for formatting
    data = np.load(file_path)
    for j in range(data_size_per_g):
        dfmax, dfmin = data[:,:,j].max(), data[:,:,j].min()
        data_nor[:,:,j] = np.array((data[:,:,j] - dfmin)/(dfmax - dfmin),dtype=float)
    data_tensor = np.concatenate([data_tensor,data_nor],axis=2)
    
np.save('./datasets_total.npy',data_tensor)

label = np.zeros((num_gestures,data_size))

for i in range(1, num_gestures+1):
    start_index = (i - 1) * data_size_per_g
    end_index = i * data_size_per_g
    label[i-1, start_index:end_index] = 1

np.save('./label.npy',label)

#%% wab test gen

data_size_per_g_test = 100
data_size_test = 1000

data_tensor_test = np.zeros((img_size,img_size,0))
data_nor_test = np.zeros((img_size,img_size,data_size_per_g_test))

for i in range(1,num_gestures+1):
    file_path = f"./datasets/{i}_test.npy"  # Using f-string for formatting
    data = np.load(file_path)
    for j in range(data_size_per_g_test):
        dfmax, dfmin = data[:,:,j].max(), data[:,:,j].min()
        data_nor_test[:,:,j] = np.array((data[:,:,j] - dfmin)/(dfmax - dfmin),dtype=float)
    data_tensor_test = np.concatenate([data_tensor_test,data_nor_test],axis=2)
    
np.save('./datasets_test.npy',data_tensor_test)

label_test = np.zeros((num_gestures,data_size_test))

for i in range(1, num_gestures+1):
    start_index = (i - 1) * 100
    end_index = i * 100
    label_test[i-1, start_index:end_index] = 1

np.save('./label_test.npy',label_test)

#%% woab test
import numpy as np

img_size = 25
num_gestures = 10
data_size_per_g = 100
data_size = 1000

data_tensor = np.zeros((img_size,img_size,0))
data_nor = np.zeros((img_size,img_size,data_size_per_g))

for i in range(1,num_gestures+1):
    file_path = f"./datasets/{i}_woab_t.npy"  # Using f-string for formatting
    data = np.load(file_path)
    for j in range(data_size_per_g):
        dfmax, dfmin = data[:,:,j].max(), data[:,:,j].min()
        data_nor[:,:,j] = np.array((data[:,:,j] - dfmin)/(dfmax - dfmin),dtype=float)
    data_tensor = np.concatenate([data_tensor,data_nor],axis=2)
    
np.save('./datasets_total_woab.npy',data_tensor)

label = np.zeros((num_gestures,data_size))

for i in range(1, num_gestures+1):
    start_index = (i - 1) * data_size_per_g
    end_index = i * data_size_per_g
    label[i-1, start_index:end_index] = 1

np.save('./label_woab.npy',label)
