# -*- coding: utf-8 -*-
"""
Boxplots accuracy Raiders dataset
"""

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.pyplot import text

#Plot VT Raiders
data =np.load('C:/Users/Angela Andreella/Documents/Thesis_Doc/Hyperaligment/Computation/MovieAnalysis/Output/perm.npz')
data1 =np.load('C:/Users/Angela Andreella/Documents/Thesis_Doc/Hyperaligment/Computation/MovieAnalysis/Output/Final_analysis_VT_cv.npz')
plt.figure(figsize=(18,18))
#plt.rcParams["axes.labelsize"] = 10
#plt.rcParams["axes.labelsize"] = 10
plt.tight_layout()
ax = sns.boxplot(data = 1-data['mean_perm'],width=0.2,  linewidth=3, fliersize = 2, color = "lightgrey")
plt.axhline(y=1-np.mean(data1["errE"]), color='red', linestyle='-',linewidth=3)
text(0.5, 1-np.mean(data1["errE"])-0.0035, "von Mises-Fisher-Procrustes model",color='red', horizontalalignment = "right",fontsize=45)
plt.axhline(y=1-np.mean(data1["gpa"]), color='green', linestyle='-',linewidth=3)
text(0.5, 1-np.mean(data1["gpa"])-0.0035, "GPA",color='green', horizontalalignment = "right",fontsize=45)
ax.set_ylabel('Mean accuracy', fontsize=45) 
ax.tick_params(labelsize=45)
ax.set_title('VT', fontsize = 45)
plt.xticks([])
plt.savefig('C:/Users/Angela Andreella/Documents/Thesis_Doc/paper_priorGPA/Plot/perm_VT.pdf')

#Plot EV Raiders
data =np.load('C:/Users/Angela Andreella/Documents/Thesis_Doc/Hyperaligment/Computation/MovieAnalysis/Output/perm_EV.npz')
data1 =np.load('C:/Users/Angela Andreella/Documents/Thesis_Doc/Hyperaligment/Computation/MovieAnalysis/Output/Final_analysis_EV_cv.npz')

plt.figure(figsize=(18,18))
#plt.rcParams["axes.labelsize"] = 10
#plt.rcParams["axes.labelsize"] = 10
plt.tight_layout()
ax = sns.boxplot(data = 1-data['mean_perm'],width=0.2,  linewidth=3, fliersize = 2, color = "lightgrey")
plt.axhline(y=1-np.mean(data1["errE"]), color='red', linestyle='-',linewidth=3)
text(0.5, 1-np.mean(data1["errE"])+0.001, "von Mises-Fisher-Procrustes model",color='red', horizontalalignment = "right",fontsize=45)
plt.axhline(y=1-np.mean(data1["gpa"]), color='green', linestyle='-',linewidth=3)
text(0.5, 1-np.mean(data1["gpa"])-0.0035, "GPA",color='green', horizontalalignment = "right",fontsize=45)
ax.set_ylabel('          ', fontsize=45) 
ax.tick_params(labelsize=45)
ax.set_title('EV', fontsize = 45)
plt.xticks([])
plt.savefig('C:/Users/Angela Andreella/Documents/Thesis_Doc/paper_priorGPA/Plot/perm_EV.pdf')

#Plot LO Raiders
data =np.load('C:/Users/Angela Andreella/Documents/Thesis_Doc/Hyperaligment/Computation/MovieAnalysis/Output/perm_LO.npz')
data1 =np.load('C:/Users/Angela Andreella/Documents/Thesis_Doc/Hyperaligment/Computation/MovieAnalysis/Output/Final_analysis_LO_cv.npz')

plt.figure(figsize=(18,18))
#plt.rcParams["axes.labelsize"] = 10
#plt.rcParams["axes.labelsize"] = 10
plt.tight_layout()
ax = sns.boxplot(data = 1-data['mean_perm'],width=0.2,  linewidth=3, fliersize = 2, color = "lightgrey")
plt.axhline(y=1-np.mean(data1["errE"]), color='red', linestyle='-',linewidth=3)
text(0.5, 1-np.mean(data1["errE"])+0.001, "von Mises-Fisher-Procrustes model",color='red', horizontalalignment = "right",fontsize=45)
plt.axhline(y=1-np.mean(data1["gpa"]), color='green', linestyle='-',linewidth=3)
text(0.5, 1-np.mean(data1["gpa"])-0.0035, "GPA",color='green', horizontalalignment = "right",fontsize=45)
ax.set_ylabel('          ', fontsize=45) 
ax.tick_params(labelsize=45)
ax.set_title('LO', fontsize = 45)
plt.xticks([])
plt.savefig('C:/Users/Angela Andreella/Documents/Thesis_Doc/paper_priorGPA/Plot/perm_LO.pdf')

##############################OHBM abstract##############################

data =np.load('C:/Users/Angela Andreella/Documents/Thesis_Doc/Hyperaligment/Computation/MovieAnalysis/Output/perm.npz')
data1 =np.load('C:/Users/Angela Andreella/Documents/Thesis_Doc/Hyperaligment/Computation/MovieAnalysis/Output/Final_analysis_VT_cv2.npz')
1-np.mean(data1["an"])
plt.figure(figsize=(28,18))
#plt.rcParams["axes.labelsize"] = 10
#plt.rcParams["axes.labelsize"] = 10
plt.tight_layout()
ax = sns.boxplot(data = 1-data['mean_perm'],width=0.2,  linewidth=3, fliersize = 2)
text(+0.49, 0.45+0.001, "Hyperalignment",color='blue', horizontalalignment = "right",fontsize=55) 

plt.axhline(y=1-np.mean(data1["errE"]), color='red', linestyle='-',linewidth=3)
text(0.5, 1-np.mean(data1["errE"])-0.0035, "Generalized Procrustes with prior",color='red', horizontalalignment = "right",fontsize=55)
plt.axhline(y=1-np.mean(data1["gpa"]), color='green', linestyle='-',linewidth=3)
text(0.5, 1-np.mean(data1["gpa"])-0.0035, "Generalized Procrustes",color='green', horizontalalignment = "right",fontsize=55)
ax.set_ylabel('Mean accuracy', fontsize=55, labelpad=30) 
ax.set_xlabel('31 subjects, Raiders movie stimuli (Haxby et al., 2011)', fontsize=55, labelpad=30) 
ax.tick_params(labelsize=55)
ax.set_title('Ventral Temporal Cortex', fontsize = 55, pad=30)
plt.xticks([])
plt.savefig('C:/Users/Angela Andreella/Documents/Conference_Talk/OHBM_2020/Plot/PlotmovieVT.pdf')
