# -*- coding: utf-8 -*-
"""
Boxplots accuracy Raiders dataset
"""

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.pyplot import text

#Plot VT Raiders
data =np.load('Analysis_VT.npz.npz')

plt.figure(figsize=(18,18))
#plt.rcParams["axes.labelsize"] = 10
#plt.rcParams["axes.labelsize"] = 10
plt.tight_layout()
ax = sns.boxplot(data = 1-data['mean_perm'],width=0.2,  linewidth=3, fliersize = 2, color = "lightgrey")
plt.axhline(y=1-np.mean(data["errE"]), color='red', linestyle='-',linewidth=3)
text(0.5, 1-np.mean(data["errE"])-0.0035, "von Mises-Fisher-Procrustes model",color='red', horizontalalignment = "right",fontsize=45)
plt.axhline(y=1-np.mean(data["gpa"]), color='green', linestyle='-',linewidth=3)
text(0.5, 1-np.mean(data["gpa"])-0.0035, "GPA",color='green', horizontalalignment = "right",fontsize=45)
ax.set_ylabel('Mean accuracy', fontsize=45) 
ax.tick_params(labelsize=45)
ax.set_title('VT', fontsize = 45)
plt.xticks([])
plt.savefig('perm_VT.pdf')

#Plot EV Raiders
data =np.load('Analysis_EV.npz.npz')

plt.figure(figsize=(18,18))
#plt.rcParams["axes.labelsize"] = 10
#plt.rcParams["axes.labelsize"] = 10
plt.tight_layout()
ax = sns.boxplot(data = 1-data['mean_perm'],width=0.2,  linewidth=3, fliersize = 2, color = "lightgrey")
plt.axhline(y=1-np.mean(data["errE"]), color='red', linestyle='-',linewidth=3)
text(0.5, 1-np.mean(data["errE"])-0.0035, "von Mises-Fisher-Procrustes model",color='red', horizontalalignment = "right",fontsize=45)
plt.axhline(y=1-np.mean(data["gpa"]), color='green', linestyle='-',linewidth=3)
text(0.5, 1-np.mean(data["gpa"])-0.0035, "GPA",color='green', horizontalalignment = "right",fontsize=45)
ax.set_ylabel('Mean accuracy', fontsize=45) 
ax.tick_params(labelsize=45)
ax.set_title('VT', fontsize = 45)
plt.xticks([])
plt.savefig('perm_EV.pdf')

#Plot LO Raiders
data =np.load('Analysis_LO.npz.npz')

plt.figure(figsize=(18,18))
#plt.rcParams["axes.labelsize"] = 10
#plt.rcParams["axes.labelsize"] = 10
plt.tight_layout()
ax = sns.boxplot(data = 1-data['mean_perm'],width=0.2,  linewidth=3, fliersize = 2, color = "lightgrey")
plt.axhline(y=1-np.mean(data["errE"]), color='red', linestyle='-',linewidth=3)
text(0.5, 1-np.mean(data["errE"])-0.0035, "von Mises-Fisher-Procrustes model",color='red', horizontalalignment = "right",fontsize=45)
plt.axhline(y=1-np.mean(data["gpa"]), color='green', linestyle='-',linewidth=3)
text(0.5, 1-np.mean(data["gpa"])-0.0035, "GPA",color='green', horizontalalignment = "right",fontsize=45)
ax.set_ylabel('Mean accuracy', fontsize=45) 
ax.tick_params(labelsize=45)
ax.set_title('VT', fontsize = 45)
plt.xticks([])
plt.savefig('perm_LO.pdf')
