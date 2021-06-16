"""
Boxplots accuracy Object-Faces datasets
"""

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.pyplot import text


#100 Voxels
data = np.load('out_hyp.npz')
data1 = np.load('out_gpa.npz')
data2 = np.load('out_ProMisesModel.npz')

plt.figure(figsize=(23,23))
#plt.rcParams["axes.labelsize"] = 10
#plt.rcParams["axes.labelsize"] = 10
plt.tight_layout()
ax = sns.boxplot(data = data['out'],width=0.2,  linewidth=5, fliersize = 2, color = "lightgrey")
plt.axhline(y=data2["mean_gpaE_results"], color='red', linestyle='-',linewidth=5)

text(+0.495, data2["mean_gpaE_results"]+0.001, "von Mises-Fisher-Procrustes model",color='red', horizontalalignment = "right",fontsize=64) 
plt.axhline(y=data1["mean_gpa0_results"], color='green', linestyle='-',linewidth=5)
text(+0.49, data1["mean_gpa0_results"]+0.001, "GPA",color='green', horizontalalignment = "right",fontsize=64) 
ax.set_ylabel('Mean accuracy', fontsize=55, labelpad=28) 
ax.tick_params(labelsize=52)
plt.xticks([])
ax.set_title('100 voxels', fontsize = 54)

plt.savefig('perm100.pdf')

