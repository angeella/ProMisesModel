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

in_path = "C:/Users/Angela Andreella/Documents/GitHub/vMFPmodel/Data/Auditory"

mask = nib.load(in_path + "/mask_Superior_Temporal_Gyrus.nii.gz")
maskdata = mask.get_fdata()
maskdata = np.reshape(maskdata, (902629))

########################################################################################
#####################################GPA################################################
########################################################################################

#10233 n voxels with mask
#902629 n voxels without mask
scores = np.zeros((10233,18)) 

for i in range(1, 18):
    img = nib.load(in_path + "/Zstat_aligned/GPA/sub-" + str(i) + "zstat3.nii.gz").get_fdata()
    img = np.reshape(img, (902629))[maskdata==1]
    scores[:,i] = img
  
scores.shape    
n = scores.shape[1]

Tstat = scores.mean(axis=1, dtype =np.float64)/(scores.std(axis=1, dtype =np.float64, ddof=0) +1e-12 /np.sqrt(n))

Pvalues = 2*(1-stats.norm.cdf(np.abs(Tstat))) 
Pvalues_gpa = Pvalues

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
                           title = "GPA Vocal  - NoVocal",
                           output_file = in_path + "zstat3_gpa.pdf",
                           cut_coords = [-55,-30,-4])


Tstat_hyp = Tstat

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
                           output_file = in_path + "zstat3Tstat_gpa.pdf",
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
                           output_file = in_path + "zstat3_gpa_rand.pdf",
                           cut_coords = [-55,-30,-4])


########################################################################################
#####################################HYPER##############################################
########################################################################################

#10233 n voxels with mask
#902629 n voxels without mask
scores = np.zeros((10233,18)) 

for i in range(1, 18):
    img = nib.load(in_path + "/Zstat_aligned/Hyper/sub-" + str(i) + "zstat3.nii.gz").get_fdata()
    img = np.reshape(img, (902629))[maskdata==1]
    scores[:,i] = img
  
scores.shape    
n = scores.shape[1]

Tstat = scores.mean(axis=1, dtype =np.float64)/(scores.std(axis=1, dtype =np.float64, ddof=0) +1e-12 /np.sqrt(n))

Pvalues = 2*(1-stats.norm.cdf(np.abs(Tstat))) 
Pvalues_hyp = Pvalues

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
                           output_file = in_path + "zstat3_hyp.pdf",
                           cut_coords = [-55,-30,-4])

#Tstat

Tstat_hyp = Tstat

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
                           output_file = "zstat3Tstat_hyp.pdf",
                           cut_coords = [-55,-30,-4])

########################################################################################
#####################################GPA PRIOR##############################################
########################################################################################

#10233 n voxels with mask
#902629 n voxels without mask
scores = np.zeros((10233,18)) 

for i in range(1, 18):
    img = nib.load(in_path + "Zstat_aligned/GPAprior/sub-" + str(i) + "zstat3.nii.gz").get_fdata()
    img = np.reshape(img, (902629))[maskdata==1]
    scores[:,i] = img
  
scores.shape    
n = scores.shape[1]

Tstat = scores.mean(axis=1, dtype =np.float64)/(scores.std(axis=1, dtype =np.float64, ddof=0) +1e-12 /np.sqrt(n))

Pvalues = 2*(1-stats.norm.cdf(np.abs(Tstat))) 
Pvalues_gpaPrior = Pvalues

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
                           output_file = "zstat3_vMFPmodel.pdf",
                           cut_coords = [-55,-30,-4])

#Tstat
Tstat_gpaPrior = Tstat
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
                           output_file = "zstat3Tstat_vMFPmodel.pdf",
                           cut_coords = [-55,-30,-4])

############################NOALIGNMENT#########################################

scores = np.zeros((10233,18)) 

for i in range(1, 18):
    img = nib.load(in_path + "Zstat_aligned/NoAlign/sub-" + str(i) + "zstat3.nii.gz").get_fdata()
    img = np.reshape(img, (902629))[maskdata==1]
    scores[:,i] = img

scores.shape    
n = scores.shape[1]

Tstat = scores.mean(axis=1, dtype =np.float64)/(scores.std(axis=1, dtype =np.float64, ddof=0) +1e-12 /np.sqrt(n))

Pvalues = 2*(1-stats.norm.cdf(np.abs(Tstat))) 
Pvalues_real = Pvalues

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
                           output_file = "zstat3_noAlignment.pdf",
                           cut_coords = [-55,-30,-4])

#Tstat
Tstat_real = Tstat
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
                           output_file = "zstat3Tstat_noAlignment.pdf",
                           cut_coords = [-55,-30,-4])


##########################################PLOT log(p)##################################################

import matplotlib.pyplot as plt
import matplotlib.lines as mlines

fig, ax = plt.subplots()
plt.scatter(-np.log10(Pvalues_hyp),-np.log10(Pvalues_gpaPrior))
line = mlines.Line2D([0, 1], [0, 1], color='red')
transform = ax.transAxes
line.set_transform(transform)
ax.add_line(line)
plt.xlabel('Hyperalignment', fontsize=18)
plt.ylabel('Fisher-Procrustes model', fontsize=16)
plt.savefig("pvalues_hyp.pdf")
plt.show()

fig, ax = plt.subplots()
plt.scatter(-np.log10(Pvalues_gp),-np.log10(Pvalues_gpaPrior))
line = mlines.Line2D([0, 1], [0, 1], color='red')
transform = ax.transAxes
line.set_transform(transform)
ax.add_line(line)
plt.xlabel('GPA', fontsize=18)
plt.ylabel('Fisher-Procrustes model', fontsize=16)
plt.savefig("pvalues_GPA.pdf")
plt.show()

fig, ax = plt.subplots()
plt.scatter(-np.log10(Pvalues_real),-np.log10(Pvalues_gpaPrior))
line = mlines.Line2D([0, 1], [0, 1], color='red')
transform = ax.transAxes
line.set_transform(transform)
ax.add_line(line)
plt.xlabel('No Alignment', fontsize=18)
plt.ylabel('Fisher-Procrustes model', fontsize=16)
plt.savefig("pvalues_real.pdf")
plt.show()

fig, ax = plt.subplots()
plt.scatter(-np.log10(Pvalues_real),-np.log10(Pvalues_hyp))
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
plt.scatter(abs(Tstat_hyp),abs(Tstat_gpaPrior))
 # set axes range
plt.xlim((0,2.5))
plt.ylim((0,2.5))
line = mlines.Line2D([0, 1], [0, 1], color='red')
transform = ax.transAxes
line.set_transform(transform)
ax.add_line(line)
plt.xlabel('Hyperalignment', fontsize=18)
plt.ylabel('von Mises-Fisher Procrustes model', fontsize=18)
plt.savefig("Tstat_hyp.pdf")
plt.show()

fig, ax = plt.subplots()
ax.set_aspect(1)
plt.scatter(abs(Tstat_gpa),abs(Tstat_gpaPrior))
plt.xlim((0,2.5))
plt.ylim((0,2.5))
line = mlines.Line2D([0, 1], [0, 1], color='red')
transform = ax.transAxes
line.set_transform(transform)
ax.add_line(line)
plt.xlabel('GPA', fontsize=18)
plt.ylabel('von Mises-Fisher Procrustes model', fontsize=18)
plt.savefig("Tstat_GPA.pdf")
plt.show()

fig, ax = plt.subplots()
ax.set_aspect(1)
plt.scatter(abs(Tstat_real),abs(Tstat_gpaPrior))
plt.xlim((0,2.5))
plt.ylim((0,2.5))
line = mlines.Line2D([0, 1], [0, 1], color='red')
transform = ax.transAxes
line.set_transform(transform)
ax.add_line(line)
plt.xlabel('No Alignment', fontsize=18)
plt.ylabel('von Mises-Fisher Procrustes model', fontsize=18)
plt.savefig("Tstat_real.pdf")
plt.show()


