#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 15:59:15 2024

@author: apple
"""
import numpy as np
import matplotlib.pyplot as plt
from utils import poisson_encode
import torch

image = np.load('datasets_test.npy')
one_img = image[:,:,100]

img_encoded = poisson_encode(torch.tensor(one_img),8)
img_encoded = img_encoded.sum(0)


plt.imshow(img_encoded, cmap='rainbow')
plt.colorbar()  # Add a colorbar for reference
plt.show()


