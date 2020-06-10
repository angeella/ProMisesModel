"""
Boxplots accuracy Object-Faces datasets
"""

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.pyplot import text


#100 Voxels
data = np.load('C:/Users/Angela Andreella/Desktop/ObjectAnalysis/Output/fs_cv_100_1.npz')
data1 = np.load('C:/Users/Angela Andreella/Desktop/ObjectAnalysis/Output/fs_cv_100_2.npz')
data2 = np.load('C:/Users/Angela Andreella/Desktop/ObjectAnalysis/Output/fs_cv_100_3.npz')

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

plt.savefig('C:/Users/Angela Andreella/Documents/Thesis_Doc/paper_priorGPA/Plot/perm100.pdf')

#200 Voxels
data = np.load('C:/Users/Angela Andreella/Desktop/ObjectAnalysis/Output/fs_cv_200_1.npz')
data2 = np.load('C:/Users/Angela Andreella/Desktop/ObjectAnalysis/Output/fs_cv_200_3.npz')
data1 =np.load('C:/Users/Angela Andreella/Documents/Thesis_Doc/Hyperaligment/Computation/ObjectAnalysis/Output/final_analysis_fs_cv_200v.npz')

plt.figure(figsize=(18,18))
#plt.rcParams["axes.labelsize"] = 10
#plt.rcParams["axes.labelsize"] = 10
plt.tight_layout()
ax = sns.boxplot(data = data['out'],width=0.2,  linewidth=3, fliersize = 2, color = "lightgrey")
plt.axhline(y=data2["mean_gpaE_results"], color='red', linestyle='-',linewidth=3)

text(+0.495, data2["mean_gpaE_results"]-0.004, "von Mises-Fisher-Procrustes model",color='red', horizontalalignment = "right",fontsize=55) 
plt.axhline(y=data1["mean_gpa0_results"], color='green', linestyle='-',linewidth=3)
text(+0.49, data1["mean_gpa0_results"]+0.001, "GPA",color='green', horizontalalignment = "right",fontsize=55) 
#ax.set_ylabel('Mean accuracy', fontsize=55, labelpad=30) 
ax.tick_params(labelsize=45)
plt.xticks([])
ax.set_title('200 voxels', fontsize = 45)

plt.savefig('C:/Users/Angela Andreella/Documents/Thesis_Doc/paper_priorGPA/Plot/perm200.pdf')

#300 Voxels
data = np.load('C:/Users/Angela Andreella/Desktop/ObjectAnalysis/Output/fs_cv_300_1.npz')
data2 =np.load('C:/Users/Angela Andreella/Desktop/ObjectAnalysis/Output/fs_cv_300_3.npz')
data1 = np.load('C:/Users/Angela Andreella/Desktop/ObjectAnalysis/Output/fs_cv_300_2.npz')

plt.figure(figsize=(18,18))
#plt.rcParams["axes.labelsize"] = 10
#plt.rcParams["axes.labelsize"] = 10
plt.tight_layout()
ax = sns.boxplot(data = data['out'],width=0.2,  linewidth=3, fliersize = 2, color = "lightgrey")
plt.axhline(y=data2["mean_gpaE_results"], color='red', linestyle='-',linewidth=3)
text(+0.495, data2["mean_gpaE_results"]-0.003, "von Mises-Fisher-Procrustes model",color='red', horizontalalignment = "right",fontsize=55) 
plt.axhline(y=data1["mean_gpa0_results"], color='green', linestyle='-',linewidth=3)
text(+0.49, data1["mean_gpa0_results"]+0.001, "GPA",color='green', horizontalalignment = "right",fontsize=55) 
ax.set_ylabel('           ', fontsize=45) 
ax.tick_params(labelsize=45)
ax.set_title('300 voxels', fontsize = 45)
plt.xticks([])

plt.savefig('C:/Users/Angela Andreella/Documents/Thesis_Doc/paper_priorGPA/Plot/perm300.pdf')

#400 Voxels
data = np.load('C:/Users/Angela Andreella/Desktop/ObjectAnalysis/Output/fs_cv_400_1.npz')
data2 =np.load('C:/Users/Angela Andreella/Desktop/ObjectAnalysis/Output/fs_cv_400_3.npz')
data1 = np.load('C:/Users/Angela Andreella/Desktop/ObjectAnalysis/Output/fs_cv_400_2.npz')

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
ax.set_title('400 voxels', fontsize = 54)

plt.savefig('C:/Users/Angela Andreella/Documents/Thesis_Doc/paper_priorGPA/Plot/perm400.pdf')


#500 Voxels
data = np.load('C:/Users/Angela Andreella/Desktop/ObjectAnalysis/Output/fs_cv_500_1.npz')
data2 =np.load('C:/Users/Angela Andreella/Desktop/ObjectAnalysis/Output/fs_cv_500_3.npz')
data1 = np.load('C:/Users/Angela Andreella/Desktop/ObjectAnalysis/Output/fs_cv_500_2.npz')

plt.figure(figsize=(18,18))
#plt.rcParams["axes.labelsize"] = 10
#plt.rcParams["axes.labelsize"] = 10
plt.tight_layout()
ax = sns.boxplot(data = data['out'],width=0.2,  linewidth=3, fliersize = 2, color = "lightgrey")
plt.axhline(y=data2["mean_gpaE_results"], color='red', linestyle='-',linewidth=3)
text(+0.495, data2["mean_gpaE_results"]-0.003, "von Mises-Fisher-Procrustes model",color='red', horizontalalignment = "right",fontsize=55) 
plt.axhline(y=data1["mean_gpa0_results"], color='green', linestyle='-',linewidth=3)
text(+0.49, data1["mean_gpa0_results"]+0.001, "GPA",color='green', horizontalalignment = "right",fontsize=55) 
ax.set_ylabel('           ', fontsize=45) 
ax.tick_params(labelsize=45)
ax.set_title('500 voxels', fontsize = 45)
plt.xticks([])

plt.savefig('C:/Users/Angela Andreella/Documents/Thesis_Doc/paper_priorGPA/Plot/perm500.pdf')

#600 Voxels
data = np.load('C:/Users/Angela Andreella/Desktop/ObjectAnalysis/Output/fs_cv_600_1.npz')
data2 =np.load('C:/Users/Angela Andreella/Desktop/ObjectAnalysis/Output/fs_cv_600_3.npz')
data1 = np.load('C:/Users/Angela Andreella/Desktop/ObjectAnalysis/Output/fs_cv_600_2.npz')

plt.figure(figsize=(18,18))
#plt.rcParams["axes.labelsize"] = 10
#plt.rcParams["axes.labelsize"] = 10
plt.tight_layout()
ax = sns.boxplot(data = data['out'],width=0.2,  linewidth=3, fliersize = 2, color = "lightgrey")
plt.axhline(y=data2["mean_gpaE_results"], color='red', linestyle='-',linewidth=3)
text(+0.495, data2["mean_gpaE_results"]-0.003, "von Mises-Fisher-Procrustes model",color='red', horizontalalignment = "right",fontsize=55) 
plt.axhline(y=data1["mean_gpa0_results"], color='green', linestyle='-',linewidth=3)
text(+0.49, data1["mean_gpa0_results"]-0.003, "GPA",color='green', horizontalalignment = "right",fontsize=55) 
ax.set_ylabel('           ', fontsize=45) 
ax.tick_params(labelsize=45)
ax.set_title('600 voxels', fontsize = 45)
plt.xticks([])

plt.savefig('C:/Users/Angela Andreella/Documents/Thesis_Doc/paper_priorGPA/Plot/perm600.pdf')