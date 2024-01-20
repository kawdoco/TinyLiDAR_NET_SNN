
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
from LiDAR_NET_SNN import LiDAR_NET_SCNN,LiDAR_NET_SNN_MLP
from utils import poisson_encode,plot_confusion_matrix
from spikingjelly.activation_based.functional import reset_net
#%%
torch.set_printoptions(precision=10)

no_samples=1000
no_gestures=10
timestep=8

frame=np.load(r'datasets_test.npy')  
gesture_gt=np.load('label_test.npy')   
gesture_gt=np.transpose(gesture_gt,(1,0))

frame=frame.astype(np.float32)

frame = Variable(torch.from_numpy(frame))
frame = frame.unsqueeze(0)
frame=np.transpose(frame, (3,0,1,2))

#use gpu
PATH=r'./LiDAR_NET_Pre_trained_model_SNN_CNN/ckpt_epoch_70_val_loss_1.463720.pth'
model = LiDAR_NET_SCNN()

#use cpu
#checkpoint = torch.load(PATH, map_location='cpu')
#model.load_state_dict(checkpoint)

checkpoint=torch.load(PATH)
model.load_state_dict(checkpoint,strict=False)
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
frame_torch = frame.to(device)

gesture_hat=torch.zeros(no_samples,1)

start = time.time()
yhat=0
with torch.no_grad():
    frame_torch_encoded = poisson_encode(frame_torch,timestep)
    reset_net(model)
    for xt in frame_torch_encoded:
        yhat += model(xt.float())
    yhat = yhat.softmax(-1)
    pred = yhat.argmax(1)
end = time.time()

max_index_gt = np.zeros(no_samples)
for i in range(no_samples):
    max_index_gt[i] = np.argmax(gesture_gt[i,:])

print('------Inference Done------')
print(f"Runtime of the program is {end - start}")# -*- coding: utf-8 -*-
#%% Accuracy evaluation
from sklearn.metrics import accuracy_score, confusion_matrix

accuracy = accuracy_score(max_index_gt, pred)
cm = confusion_matrix(max_index_gt, pred, labels=np.arange(0,10))

plt.figure(figsize=(8, 6))
sns.heatmap(cm, fmt="d", cmap="Blues", cbar=True,vmin=0, vmax=100)
plt.xticks(ticks=np.arange(0,10), labels=np.arange(0,10)+1)
plt.yticks(ticks=np.arange(0,10), labels=np.arange(0,10)+1)
plt.title(f"SCNN Confusion Matrix w/o AL\nAccuracy: {accuracy*100}%")
plt.xlabel("Predicted gestures")
plt.ylabel("True gestures")
plt.tight_layout()
dpi_value = 300  # Adjust this value as needed
# plt.savefig(r'./Figures/SCNN_confusion_matrix_wo_AL.png', dpi=dpi_value)
plt.show()
#%%
index=950
encoded=frame_torch_encoded.numpy()
encoded=encoded.sum(0)
encoded=encoded.squeeze()

original_pick = frame_torch[index,:,:,:].squeeze().numpy()
encoded_pick = encoded[index,:,:]

plt.imshow(encoded_pick, cmap='rainbow')
plt.tight_layout()
#%%
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np

accuracy = accuracy_score(max_index_gt, pred)
# Assuming max_index_gt and pred are your ground truth and predicted labels
# Adjust labels parameter based on the range of your labels
labels = np.arange(0, 10)

# Create the confusion matrix
cm = confusion_matrix(max_index_gt, pred, labels=labels)
# Plot the confusion matrix as a heatmap with values in each square
plt.figure(figsize=(10, 8))
plot_confusion_matrix(cm, classes=labels, normalize=True, savename="Figures/SCNN_confusion_matrix.png", title=f'SCNN Confusion Matrix w/ AL \nAccuracy: {accuracy*100:.2f}%')

