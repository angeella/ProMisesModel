#!/usr/bin/env python
"""
Maps p values

@author: Angela Andreella
"""
import nibabel as nib
from nilearn import plotting
import numpy as np
from scipy import stats
import matplotlib as mpl
#Read data

in_path = "C:/Users/Angela Andreella/Documents/Thesis_Doc/Hyperaligment/Computation/AuditoryData/"

mask = nib.load(in_path + "Pvalues/STG/mask_Superior_Temporal_Gyrus.nii.gz")
maskdata = mask.get_fdata()
maskdata = np.reshape(maskdata, (902629))

########################################################################################
#####################################GPA################################################
########################################################################################

#10233 n voxels with mask
#902629 n voxels without mask
scores = np.zeros((10233,18)) 

for i in range(1, 18):
    img = nib.load(in_path + "Pvalues/STG/Vocal/GPA/sub-" + str(i) + "zstat1.nii.gz").get_fdata()
    img = np.reshape(img, (902629))[maskdata==1]
    scores[:,i] = img
  
scores.shape    
n = scores.shape[1]

Tstat = scores.mean(axis=1, dtype =np.float64)/(scores.std(axis=1, dtype =np.float64, ddof=0) +1e-12 /np.sqrt(n))

Pvalues = 2*(1-stats.norm.cdf(np.abs(Tstat))) 
Pvalues_gpa_1 = Pvalues

#Tstat = np.sum(scores,axis=1)/np.sqrt(n)

Pvalues1 = np.copy(maskdata)
j = 0
for i in range(902629):
    if Pvalues1[i] == 1:
            Pvalues1[i] = Pvalues[j]
            j = j + 1
    else:
       Pvalues1[i] = np.nan 
        

Pvalues1 = np.reshape(Pvalues1,(91,109,91))

Pvalues_img = nib.Nifti1Image(Pvalues1, affine=[[  -2.,    0.,    0.,   90.],
                                           [   0.,    2.,    0., -126.],
                                           [   0.,    0.,    2.,  -72.],
                                           [   0.,    0.,    0.,    1.]])

plotting.plot_stat_map(Pvalues_img,
                           title = "GPA Vocal",
                           output_file = in_path + "/out/zstat1_gpa.pdf",
                           cut_coords = [-59,-23,-3])

#zstat3

scores = np.zeros((10233,18)) 

for i in range(1, 18):
    img = nib.load(in_path + "Pvalues/STG/Vocal_NoVocal/GPA/sub-" + str(i) + "zstat3.nii.gz").get_fdata()
    img = np.reshape(img, (902629))[maskdata==1]
    scores[:,i] = img
  
scores.shape    
n = scores.shape[1]

Tstat = scores.mean(axis=1, dtype =np.float64)/(scores.std(axis=1, dtype =np.float64, ddof=0) +1e-12 /np.sqrt(n))

Pvalues = 2*(1-stats.norm.cdf(np.abs(Tstat))) 
Pvalues_gpa_3 = Pvalues

Pvalues_rand = np.random.permutation(Pvalues)

#Tstat = np.sum(scores,axis=1)/np.sqrt(n)

Pvalues1 = np.copy(maskdata)
j = 0
for i in range(902629):
    if Pvalues1[i] == 1:
            Pvalues1[i] = Pvalues[j]
            j = j + 1
    else:
       Pvalues1[i] = np.nan 
        

Pvalues1 = np.reshape(Pvalues1,(91,109,91))

Pvalues_img = nib.Nifti1Image(Pvalues1, affine=[[  -2.,    0.,    0.,   90.],
                                           [   0.,    2.,    0., -126.],
                                           [   0.,    0.,    2.,  -72.],
                                           [   0.,    0.,    0.,    1.]])

plotting.plot_stat_map(Pvalues_img,
                           title = "GPA Vocal - NoVocal",
                           output_file = in_path + "/out/zstat3_gpa.pdf",
                           cut_coords = [-55,-30,-4])


Pvalues1 = np.copy(maskdata)
j = 0
for i in range(902629):
    if Pvalues1[i] == 1:
            Pvalues1[i] = Pvalues_rand[j]
            j = j + 1
    else:
       Pvalues1[i] = np.nan 
        

Pvalues1 = np.reshape(Pvalues1,(91,109,91))

Pvalues_img = nib.Nifti1Image(Pvalues1, affine=[[  -2.,    0.,    0.,   90.],
                                           [   0.,    2.,    0., -126.],
                                           [   0.,    0.,    2.,  -72.],
                                           [   0.,    0.,    0.,    1.]])
norm = mpl.colors.Normalize(vmin=0,vmax=1.)

plotting.plot_stat_map(Pvalues_img, cmap = 'ocean_hot',
                           title = "GPA Vocal - NoVocal",
                           output_file = in_path + "/out/zstat3_gpa_rand.pdf",
                           cut_coords = [-55,-30,-4])


#zstat3 Test

scores = np.zeros((10233,18)) 

for i in range(1, 18):
    img = nib.load(in_path + "Pvalues/STG/Vocal_NoVocal/GPA/sub-" + str(i) + "zstat3.nii.gz").get_fdata()
    img = np.reshape(img, (902629))[maskdata==1]
    scores[:,i] = img
  
scores.shape    
n = scores.shape[1]

Tstat = scores.mean(axis=1, dtype =np.float64)/(scores.std(axis=1, dtype =np.float64, ddof=0) +1e-12 /np.sqrt(n))

Tstat_gpa_3 = Tstat

Tstat_rand = np.random.permutation(Tstat)

#Tstat = np.sum(scores,axis=1)/np.sqrt(n)

Tstat1 = np.copy(maskdata)
j = 0
for i in range(902629):
    if Tstat1[i] == 1:
            Tstat1[i] = Tstat[j]
            j = j + 1
    else:
       Tstat1[i] = np.nan 
        

Tstat1 = np.reshape(Tstat1,(91,109,91))

Tstat_img = nib.Nifti1Image(Tstat1, affine=[[  -2.,    0.,    0.,   90.],
                                           [   0.,    2.,    0., -126.],
                                           [   0.,    0.,    2.,  -72.],
                                           [   0.,    0.,    0.,    1.]])

plotting.plot_stat_map(Tstat_img,
                           title = "GPA Vocal - NoVocal",
                           output_file = in_path + "/out/zstat3Tstat_gpa.pdf",
                           cut_coords = [-55,-30,-4])


Tstat_rand1 = np.copy(maskdata)
j = 0
for i in range(902629):
    if Tstat_rand1[i] == 1:
            Tstat_rand1[i] = Tstat_rand[j]
            j = j + 1
    else:
       Tstat_rand1[i] = np.nan 
        

Tstat_rand1 = np.reshape(Tstat_rand1,(91,109,91))

Tstat_rand_img = nib.Nifti1Image(Tstat_rand1, affine=[[  -2.,    0.,    0.,   90.],
                                           [   0.,    2.,    0., -126.],
                                           [   0.,    0.,    2.,  -72.],
                                           [   0.,    0.,    0.,    1.]])
norm = mpl.colors.Normalize(vmin=0,vmax=1.)

plotting.plot_stat_map(Tstat_rand_img, 
                           title = "GPA Vocal - NoVocal",
                           output_file = in_path + "/out/zstat3Tstat_rand_gpa.pdf",
                           cut_coords = [-55,-30,-4])

########################################################################################
#####################################HYPER##############################################
########################################################################################

#10233 n voxels with mask
#902629 n voxels without mask
scores = np.zeros((10233,18)) 

for i in range(1, 18):
    img = nib.load(in_path + "Pvalues/STG/Vocal/Hyper/sub-" + str(i) + "zstat1.nii.gz").get_fdata()
    img = np.reshape(img, (902629))[maskdata==1]
    scores[:,i] = img
  
scores.shape    
n = scores.shape[1]

Tstat = scores.mean(axis=1, dtype =np.float64)/(scores.std(axis=1, dtype =np.float64, ddof=0) +1e-12 /np.sqrt(n))

Pvalues = 2*(1-stats.norm.cdf(np.abs(Tstat))) 
Pvalues_hyp_1 = Pvalues

#Tstat = np.sum(scores,axis=1)/np.sqrt(n)

Pvalues1 = np.copy(maskdata)
j = 0
for i in range(902629):
    if Pvalues1[i] == 1:
            Pvalues1[i] = Pvalues[j]
            j = j + 1
    else:
       Pvalues1[i] = np.nan 
        

Pvalues1 = np.reshape(Pvalues1,(91,109,91))

Pvalues_img = nib.Nifti1Image(Pvalues1, affine=[[  -2.,    0.,    0.,   90.],
                                           [   0.,    2.,    0., -126.],
                                           [   0.,    0.,    2.,  -72.],
                                           [   0.,    0.,    0.,    1.]])

plotting.plot_stat_map(Pvalues_img,colorbar=True,
                           title = "Hyperalignment Vocal",
                           output_file = in_path + "/out/zstat1_hyp.pdf",
                           cut_coords = [-59,-23,-3])

#zstat3

scores = np.zeros((10233,18)) 

for i in range(1, 18):
    img = nib.load(in_path + "Pvalues/STG/Vocal_NoVocal/Hyper/sub-" + str(i) + "zstat3.nii.gz").get_fdata()
    img = np.reshape(img, (902629))[maskdata==1]
    scores[:,i] = img
  
scores.shape    
n = scores.shape[1]

Tstat = scores.mean(axis=1, dtype =np.float64)/(scores.std(axis=1, dtype =np.float64, ddof=0) +1e-12 /np.sqrt(n))

Pvalues = 2*(1-stats.norm.cdf(np.abs(Tstat))) 
Pvalues_hyp_3 = Pvalues

#Tstat = np.sum(scores,axis=1)/np.sqrt(n)

Pvalues1 = np.copy(maskdata)
j = 0
for i in range(902629):
    if Pvalues1[i] == 1:
            Pvalues1[i] = Pvalues[j]
            j = j + 1
    else:
       Pvalues1[i] = np.nan 
        

Pvalues1 = np.reshape(Pvalues1,(91,109,91))

Pvalues_img = nib.Nifti1Image(Pvalues1, affine=[[  -2.,    0.,    0.,   90.],
                                           [   0.,    2.,    0., -126.],
                                           [   0.,    0.,    2.,  -72.],
                                           [   0.,    0.,    0.,    1.]])
nib.save(Pvalues_img, in_path + '/out/Pvalues3_Hyper.nii.gz')

plotting.plot_stat_map(Pvalues_img, 
                           title = "Hyperalignment Vocal - NoVocal",
                           output_file = in_path + "/out/zstat3_hyp.pdf",
                           cut_coords = [-55,-30,-4])

#Tstat

Tstat_hyp_3 = Tstat

#Tstat = np.sum(scores,axis=1)/np.sqrt(n)

Tstat1 = np.copy(maskdata)
j = 0
for i in range(902629):
    if Tstat1[i] == 1:
            Tstat1[i] = Tstat[j]
            j = j + 1
    else:
       Tstat1[i] = np.nan 
        

Tstat1 = np.reshape(Tstat1,(91,109,91))

Tstat_img = nib.Nifti1Image(Tstat1, affine=[[  -2.,    0.,    0.,   90.],
                                           [   0.,    2.,    0., -126.],
                                           [   0.,    0.,    2.,  -72.],
                                           [   0.,    0.,    0.,    1.]])

plotting.plot_stat_map(Tstat_img, 
                           title = "Hyperalignment Vocal - NoVocal",
                           output_file = in_path + "/out/zstat3Tstat_hyp.pdf",
                           cut_coords = [-55,-30,-4])

########################################################################################
#####################################GPA PRIOR##############################################
########################################################################################

#10233 n voxels with mask
#902629 n voxels without mask

scores = np.zeros((10233,18)) 

for i in range(1, 18):
    img = nib.load(in_path + "Pvalues/STG/Vocal/GPAprior/sub-" + str(i) + "zstat1.nii.gz").get_fdata()
    img = np.reshape(img, (902629))[maskdata==1]
    scores[:,i] = img
  
scores.shape    
n = scores.shape[1]

Tstat = scores.mean(axis=1, dtype =np.float64)/(scores.std(axis=1, dtype =np.float64, ddof=0) +1e-12 /np.sqrt(n))

Pvalues = 2*(1-stats.norm.cdf(np.abs(Tstat))) 
Pvalues_gpaPrior3_1 = Pvalues

#Tstat = np.sum(scores,axis=1)/np.sqrt(n)

Pvalues1 = np.copy(maskdata)
j = 0
for i in range(902629):
    if Pvalues1[i] == 1:
            Pvalues1[i] = Pvalues[j]
            j = j + 1
    else:
       Pvalues1[i] = np.nan 
        

Pvalues1 = np.reshape(Pvalues1,(91,109,91))

Pvalues_img = nib.Nifti1Image(Pvalues1, affine=[[  -2.,    0.,    0.,   90.],
                                           [   0.,    2.,    0., -126.],
                                           [   0.,    0.,    2.,  -72.],
                                           [   0.,    0.,    0.,    1.]])

plotting.plot_stat_map(Pvalues_img,
                           title = "von Mises-Fisher Procrustes model Vocal",
                           output_file = in_path + "/out/zstat1_gpaPrior.pdf",
                           cut_coords = [-59,-23,-3])

#zstat3

scores = np.zeros((10233,18)) 

for i in range(1, 18):
    img = nib.load(in_path + "Pvalues/STG/Vocal_NoVocal/GPAprior/sub-" + str(i) + "zstat3.nii.gz").get_fdata()
    img = np.reshape(img, (902629))[maskdata==1]
    scores[:,i] = img
  
scores.shape    
n = scores.shape[1]

Tstat = scores.mean(axis=1, dtype =np.float64)/(scores.std(axis=1, dtype =np.float64, ddof=0) +1e-12 /np.sqrt(n))

Pvalues = 2*(1-stats.norm.cdf(np.abs(Tstat))) 
Pvalues_gpaPrior3_3 = Pvalues

Pvalues1 = np.copy(maskdata)
j = 0
for i in range(902629):
    if Pvalues1[i] == 1:
            Pvalues1[i] = Pvalues[j]
            j = j + 1
    else:
       Pvalues1[i] = np.nan 
        

Pvalues1 = np.reshape(Pvalues1,(91,109,91))

Pvalues_img = nib.Nifti1Image(Pvalues1, affine=[[  -2.,    0.,    0.,   90.],
                                           [   0.,    2.,    0., -126.],
                                           [   0.,    0.,    2.,  -72.],
                                           [   0.,    0.,    0.,    1.]])

    
plotting.plot_stat_map(Pvalues_img,
                           title = "von Mises-Fisher Procrustes model Vocal - NoVocal",
                           output_file = in_path + "/out/zstat3_gpaPrior.pdf",
                           cut_coords = [-55,-30,-4])



############################NOALIGNMENT#########################################

scores = np.zeros((10233,18)) 

for i in range(1, 18):
    img = nib.load(in_path + "Pvalues/STG/Vocal_NoVocal/NoAlign/sub-" + str(i) + "zstat3.nii.gz").get_fdata()
    img = np.reshape(img, (902629))[maskdata==1]
    scores[:,i] = img

scores.shape    
n = scores.shape[1]

Tstat = scores.mean(axis=1, dtype =np.float64)/(scores.std(axis=1, dtype =np.float64, ddof=0) +1e-12 /np.sqrt(n))

Pvalues = 2*(1-stats.norm.cdf(np.abs(Tstat))) 
Pvalues_real_3 = Pvalues

#Tstat = np.sum(scores,axis=1)/np.sqrt(n)

Pvalues1 = np.copy(maskdata)
j = 0
for i in range(902629):
    if Pvalues1[i] == 1:
            Pvalues1[i] = Pvalues[j]
            j = j + 1
    else:
       Pvalues1[i] = np.nan 
        

Pvalues1 = np.reshape(Pvalues1,(91,109,91))

Pvalues_img = nib.Nifti1Image(Pvalues1, affine=[[  -2.,    0.,    0.,   90.],
                                           [   0.,    2.,    0., -126.],
                                           [   0.,    0.,    2.,  -72.],
                                           [   0.,    0.,    0.,    1.]])

plotting.plot_stat_map(Pvalues_img,
                           title = "No alignment Vocal - NoVocal",
                           output_file = in_path + "/out/zstat3_noAlignment.pdf",
                           cut_coords = [-55,-30,-4])

#Tstat
Tstat_real_3 = Tstat
Tstat1 = np.copy(maskdata)
j = 0
for i in range(902629):
    if Tstat1[i] == 1:
            Tstat1[i] = Tstat[j]
            j = j + 1
    else:
       Tstat1[i] = np.nan 
        

Tstat1 = np.reshape(Tstat1,(91,109,91))

Tstat_img = nib.Nifti1Image(Tstat1, affine=[[  -2.,    0.,    0.,   90.],
                                           [   0.,    2.,    0., -126.],
                                           [   0.,    0.,    2.,  -72.],
                                           [   0.,    0.,    0.,    1.]])

plotting.plot_stat_map(Tstat_img,
                           title = "No alignment Vocal - NoVocal",
                           output_file = in_path + "/out/zstat3Tstat_noAlignment.pdf",
                           cut_coords = [-55,-30,-4])

##########################################SOME INDEX##################################################

np.mean(Pvalues_real_3> Pvalues_gpaPrior1_3) #0.8584970194468875
np.mean(Pvalues_hyp_3> Pvalues_gpaPrior1_3) #0.7029219192807583
np.mean(Pvalues_hyp_3> Pvalues_gpaPrior3_3) #0.5845793022574025
np.mean(Pvalues_gpa_3> Pvalues_gpaPrior3_3) #0.6444835336655917
np.mean(Pvalues_gpa_3> Pvalues_gpaPrior1_3) #0.8081696472197791


np.mean(abs(Tstat_hyp_3) < abs(Tstat_gpaPrior1_3)) #0.7029219192807583
np.mean(abs(Tstat_gpa_3) < abs(Tstat_gpaPrior1_3)) #0.8081696472197791
np.mean(abs(Tstat_real_3) < abs(Tstat_gpaPrior1_3)) #0.8584970194468875


##########################################PLOT log(p)##################################################

import matplotlib.pyplot as plt
import matplotlib.lines as mlines

fig, ax = plt.subplots()
plt.scatter(-np.log10(Pvalues_hyp_3),-np.log10(Pvalues_gpaPrior1_3))
line = mlines.Line2D([0, 1], [0, 1], color='red')
transform = ax.transAxes
line.set_transform(transform)
ax.add_line(line)
plt.xlabel('Hyperalignment', fontsize=18)
plt.ylabel('Fisher-Procrustes model', fontsize=16)
plt.savefig(in_path + "out/pvalues_hyp.pdf")
plt.show()

fig, ax = plt.subplots()
plt.scatter(-np.log10(Pvalues_gpa_3),-np.log10(Pvalues_gpaPrior1_3))
line = mlines.Line2D([0, 1], [0, 1], color='red')
transform = ax.transAxes
line.set_transform(transform)
ax.add_line(line)
plt.xlabel('GPA', fontsize=18)
plt.ylabel('Fisher-Procrustes model', fontsize=16)
plt.savefig(in_path + "out/pvalues_GPA.pdf")
plt.show()

fig, ax = plt.subplots()
plt.scatter(-np.log10(Pvalues_real_3),-np.log10(Pvalues_gpaPrior1_3))
line = mlines.Line2D([0, 1], [0, 1], color='red')
transform = ax.transAxes
line.set_transform(transform)
ax.add_line(line)
plt.xlabel('No Alignment', fontsize=18)
plt.ylabel('Fisher-Procrustes model', fontsize=16)
plt.savefig(in_path + "out/pvalues_real.pdf")
plt.show()

fig, ax = plt.subplots()
plt.scatter(-np.log10(Pvalues_real_3),-np.log10(Pvalues_hyp_3))
line = mlines.Line2D([0, 1], [0, 1], color='red')
transform = ax.transAxes
line.set_transform(transform)
ax.add_line(line)
plt.xlabel('No Alignment', fontsize=18)
plt.ylabel('Hyp', fontsize=16)
plt.show()


##########################################PLOT Zstat##################################################

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from pylab import rcParams

rcParams['figure.figsize'] =  [6.4, 6.4]

fig, ax = plt.subplots()
ax.set_aspect(1)
plt.scatter(abs(Tstat_hyp_3),abs(Tstat_gpaPrior1_3))
 # set axes range
plt.xlim((0,2.5))
plt.ylim((0,2.5))
line = mlines.Line2D([0, 1], [0, 1], color='red')
transform = ax.transAxes
line.set_transform(transform)
ax.add_line(line)
plt.xlabel('Hyperalignment', fontsize=18)
plt.ylabel('von Mises-Fisher Procrustes model', fontsize=18)
plt.savefig(in_path + "out/Tstat_hyp.pdf")
plt.show()

fig, ax = plt.subplots()
ax.set_aspect(1)
plt.scatter(abs(Tstat_gpa_3),abs(Tstat_gpaPrior1_3))
plt.xlim((0,2.5))
plt.ylim((0,2.5))
line = mlines.Line2D([0, 1], [0, 1], color='red')
transform = ax.transAxes
line.set_transform(transform)
ax.add_line(line)
plt.xlabel('GPA', fontsize=18)
plt.ylabel('von Mises-Fisher Procrustes model', fontsize=18)
plt.savefig(in_path + "out/Tstat_GPA.pdf")
plt.show()

fig, ax = plt.subplots()
ax.set_aspect(1)
plt.scatter(abs(Tstat_real_3),abs(Tstat_gpaPrior1_3))
plt.xlim((0,2.5))
plt.ylim((0,2.5))
line = mlines.Line2D([0, 1], [0, 1], color='red')
transform = ax.transAxes
line.set_transform(transform)
ax.add_line(line)
plt.xlabel('No Alignment', fontsize=18)
plt.ylabel('von Mises-Fisher Procrustes model', fontsize=18)
plt.savefig(in_path + "out/Tstat_real.pdf")
plt.show()


