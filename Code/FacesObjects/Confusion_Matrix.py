# -*- coding: utf-8 -*-
"""
Created on Sat Nov  2 09:33:01 2019

@author: Angela Andreella
"""
import numpy as np
import matplotlib.pyplot as plt

data = np.load('out_hyp.npz')
data1 = np.load('out_gpa.npz')
data2 = np.load('out_vMFPmodel.npz')

#cm_mean = np.mean(data["cm_mean"], axis = 0)
cm_mean = data["cm_mean"][53]

nrows = 1
ncols = 4
fig = plt.figure(figsize=(85, 85))
labels = [' Chair', ' DogFace', ' FemaleFace', ' House', ' MaleFace', ' MonkeyFace', ' Shoe']

############################Confusion matrix anatomical alignment############################
ax = fig.add_subplot(nrows, ncols, 1)
im = ax.imshow(data["cmA"])
ax.set(xticks=np.arange(data["cmA"].shape[1]),
           yticks=np.arange(data["cmA"].shape[0]),
           # ... and label them with the respective list entries
           xticklabels=labels, yticklabels=labels)
# Rotate the tick labels and set their alignment.
plt.setp(ax.get_xticklabels(), rotation=45, ha="right",rotation_mode="anchor")
plt.xticks(fontsize='80')
plt.yticks(fontsize='80')
#ax.set_xlabel('Predicted label', fontsize='18')
ax.set_ylabel('True label', fontsize='100')
ax.set_title('Anatomical\n', fontsize='100')


############################Confusion matrix hyperalignment############################

ax = fig.add_subplot(nrows, ncols, 2)
ax.imshow(cm_mean)
ax.set(xticks=np.arange(cm_mean.shape[1]),
           yticks=np.arange(cm_mean.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=labels, yticklabels=['','', '', '', '', '', ''])
# Rotate the tick labels and set their alignment.
plt.setp(ax.get_xticklabels(), rotation=45, ha="right",rotation_mode="anchor")
plt.xticks(fontsize='80')
#ax.set_xlabel('Predicted label', fontsize='18')
#fig.suptitle('Hyperalignment', fontsize='18')
ax.set_title('Hyperalignment\n', fontsize='100')
############################Confusion matrix gpa############################

ax = fig.add_subplot(nrows, ncols, 3)
ax.imshow(data1["cm0_mean"])
ax.set(xticks=np.arange(data1["cm0_mean"].shape[1]),
           yticks=np.arange(data1["cm0_mean"].shape[0]),
           # ... and label them with the respective list entries
           xticklabels=labels, yticklabels=['','', '', '', '', '', ''])
# Rotate the tick labels and set their alignment.
plt.setp(ax.get_xticklabels(), rotation=45, ha="right",rotation_mode="anchor")
plt.xticks(fontsize='80')
#ax.set_xlabel('Predicted label', fontsize='18')
#fig.suptitle('GPA', fontsize='18')
ax.set_title('GPA\n', fontsize='100')

############################Confusion matrix von Mises Fisher Procrustes model############################

ax = fig.add_subplot(nrows, ncols, 4)
im = ax.imshow(data2["cmE_mean"])
ax.set(xticks=np.arange(data2["cmE_mean"].shape[1]),
           yticks=np.arange(data2["cmE_mean"].shape[0]),
           # ... and label them with the respective list entries
           xticklabels=labels, yticklabels=['','', '', '', '', '', ''])
# Rotate the tick labels and set their alignment.
plt.setp(ax.get_xticklabels(), rotation=45, ha="right",rotation_mode="anchor")
plt.xticks(fontsize='80')
#ax.set_xlabel('Predicted label', fontsize='15')
#fig.suptitle('GPA with prior', fontsize='18')
ax.set_title('von Mises-Fisher-Procrustes model\n', fontsize='100')

fig.text(0.5, 0, 'Predicted label', ha='center', fontsize='100')
fig.subplots_adjust(bottom=-0.5)

fig.savefig('C:/Users/Angela Andreella/Documents/Thesis_Doc/paper_priorGPA/Plot/cm.pdf')
