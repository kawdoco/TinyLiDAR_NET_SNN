# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

load_curves_CNN = np.load(r'./Loss.npz')
Train_Acc_Total_CNN=load_curves_CNN['Train_Acc_Total']
Val_Acc_Total_CNN=load_curves_CNN['Val_Acc_Total']

load_curves_SCNN = np.load(r'./Loss_SNN_CNN.npz')
Train_Acc_Total_SCNN=load_curves_SCNN['Train_Acc_Total']
Val_Acc_Total_SCNN=load_curves_SCNN['Val_Acc_Total']

load_curves_SMLP = np.load(r'./Loss_SNN_MLP.npz')
Train_Acc_Total_SMLP=load_curves_SMLP['Train_Acc_Total']
Val_Acc_Total_SMLP=load_curves_SMLP['Val_Acc_Total']

plt.figure(figsize=(8, 6))

Train_Acc_Total_CNN = [x * 100 for x in Train_Acc_Total_CNN]
Val_Acc_Total_CNN = [x * 100 for x in Val_Acc_Total_CNN]

Train_Acc_Total_SCNN = [x * 100 for x in Train_Acc_Total_SCNN]
Val_Acc_Total_SCNN = [x * 100 for x in Val_Acc_Total_SCNN]

Train_Acc_Total_SMLP = [x * 100 for x in Train_Acc_Total_SMLP]
Val_Acc_Total_SMLP = [x * 100 for x in Val_Acc_Total_SMLP]

plt.plot(Train_Acc_Total_CNN, label='CNN Training Acc.', marker='o',markersize=4,color='#8ECFC9')
plt.plot(Val_Acc_Total_CNN, label='CNN Validation Acc.', marker='o',markersize=4,color='#FFBE7A')

plt.plot(Train_Acc_Total_SCNN, label='SCNN Training Acc.', marker='o',markersize=4,color='#FA7F6F')
plt.plot(Val_Acc_Total_SCNN, label='SCNN Validation Acc.', marker='o',markersize=4,color='#82B0D2')

plt.plot(Train_Acc_Total_SMLP, label='SMLP Training Acc.', marker='o',markersize=4,color='#BEB8DC')
plt.plot(Val_Acc_Total_SMLP, label='SMLP Validation Acc.', marker='o',markersize=4,color='#E7DAD2')

plt.title('Training and Validation Acc. Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy %')
plt.legend()
plt.grid(True)
plt.tight_layout()
dpi_value = 300  # Adjust this value as needed
plt.savefig(r'./Figures/acc_total.png', dpi=dpi_value)
plt.show()