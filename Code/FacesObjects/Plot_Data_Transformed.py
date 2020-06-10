# -*- coding: utf-8 -*-
"""
Alignment rotation

@author: Angela Andreella
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from scipy.spatial.distance import squareform
import seaborn as sns
import mvpa2
from mvpa2.suite import *
import scipy
from scipy.stats import ortho_group 
from nilearn import plotting

#############################################ANATOMICAL#########################################

ds_all = mvpa2.suite.h5load('//dartfs-hpc/rc/home/w/f003vpw/ObjectAnalysis/data/hyperalignment_tutorial_data_2.4.hdf5.gz')
coord = ds_all[0].fa.voxel_indices

img = map2nifti(ds_all[6], ds_all[6].samples[11,:])
affine = ds_all[1].a['imgaffine'].value
stat_img = nib.Nifti1Image(img.dataobj, affine)
plotting.plot_stat_map(stat_img, 
                       cut_coords = (-35,-52,-14),
                       title = "Anatomical",
                       output_file = "C:/Users/Angela Andreella/Documents/Thesis_Doc/paper_priorGPA/Plot/sub7_an.pdf")

for i in range(10):
    # img is now an in-memory 3D img
    img = map2nifti(ds_all[i], ds_all[i].samples[23,:])
    affine = ds_all[i].a['imgaffine'].value
    stat_img = nib.Nifti1Image(img.dataobj, affine)
    plotting.plot_stat_map(stat_img, 
                           colorbar=True,
                           cut_coords = (-40,-52,-14),
                           #vmax = 1.2,
                           title = "Anatomical",
                           output_file = "C:/Users/Angela Andreella/Documents/Thesis_Doc/Paper_GPA_Neuro - Copy/sub" + str(i) + "_an.pdf")



display = plotting.plot_glass_brain(None)
# Here, we project statistical maps with filled=True
display.add_contours(stat_img, filled = True)
display.title("Anatomical alignment")
display.output_file("C:/Users/Angela Andreella/Documents/Thesis_Doc/Paper_GPA_Neuro - Copy/Xest_an_contour.pdf")


#######################################HYPERALIGNMENT#############################################

data = np.load('C:/Users/Angela Andreella/Documents/Thesis_Doc/Hyperaligment/Computation/ObjectAnalysis/Output/al_hyp_VT.npz')
R1 = np.load('C:/Users/Angela Andreella/Documents/Thesis_Doc/Hyperaligment/Computation/R.npz')['R1']
Xest = data['Xest']

img = map2nifti(ds_all[6], Xest[6][11,:])
affine = ds_all[1].a['imgaffine'].value
stat_img = nib.Nifti1Image(img.dataobj, affine)
plotting.plot_stat_map(stat_img, 
                       cut_coords = (-35,-52,-14),
                       title = "Hyperalignment",
                       output_file = "C:/Users/Angela Andreella/Documents/Thesis_Doc/paper_priorGPA/Plot/sub7_hyp.pdf")





Xest_p = [np.dot(x,R1) for x in Xest]

ax = plt.axes(projection='3d')
sc = ax.scatter3D(x, y, z, c=Xest[8][12,:])
plt.colorbar(sc)

for i in range(10):
    # img is now an in-memory 3D img
    img = map2nifti(ds_all[i], Xest[i][23,:])
    affine = ds_all[i].a['imgaffine'].value
    stat_img = nib.Nifti1Image(img.dataobj, affine)
    plotting.plot_stat_map(stat_img, 
                           colorbar=True,
                          # cut_coords = (-35,-51,-13),
                          # vmax = 0.002,
                           title = "Hyperalignment",
                           output_file = "C:/Users/Angela Andreella/Documents/Thesis_Doc/Paper_GPA_Neuro - Copy/sub" + str(i) + "_hyp.pdf")


f = plotting.plot_glass_brain(None)
f.add_contours(stat_img, filled = True)
f.title("Hyperalignment")

ax = plt.axes(projection='3d')
sc = ax.scatter3D(x, y, z, c=Xest_p[8][12,:])
plt.colorbar(sc)

img = map2nifti(ds_all[8], Xest_p[8][23,:])
affine = ds_all[8].a['imgaffine'].value
stat_img = nib.Nifti1Image(img.dataobj, affine)
plotting.plot_stat_map(stat_img, 
                       #cut_coords = (-39,-53,-13),
                      # vmax = 1.5,
                       output_file = "C:/Users/Angela Andreella/Documents/Thesis_Doc/Paper_GPA_Neuro - Copy/Xest_hypR.pdf",
                       title = "Hyperalignment rotated")

f = plotting.plot_glass_brain(None)
f.add_contours(stat_img, filled = True)
f.title("Hyperalignment")

###########################################GPA##################################################

data = np.load('C:/Users/Angela Andreella/Documents/Thesis_Doc/Hyperaligment/Computation/Output_Object/al_gpa_VT.npz')
data.files

Xest = data['Xest']
Xest1 = data['Xest1']

R1 = ortho_group.rvs(dim=Xest[0].shape[1])
Xest_p = [np.dot(x,R1) for x in Xest]


plt.imshow(Xest[0])

x = coord[:,0]
y = coord[:,1]
z = coord[:,2]

ax = plt.axes(projection='3d')
sc = ax.scatter3D(x, y, z, c=Xest[8][50,:])
plt.colorbar(sc)

ax = plt.axes(projection='3d')
sc = ax.scatter3D(x, z, y, c=Xest1[8][20,0,:])
plt.colorbar(sc)

ax = plt.axes(projection='3d')
sc = ax.scatter3D(x, z, y, c=ds_all[3].samples[20,:])
plt.colorbar(sc)

data = np.load('C:/Users/Angela Andreella/Documents/Thesis_Doc/Hyperaligment/Computation/ObjectAnalysis/Output/al_gpa_VT.npz')
data.files
Xest = data['Xest']

img = map2nifti(ds_all[6], Xest[6][11,:])
affine = ds_all[6].a['imgaffine'].value
stat_img = nib.Nifti1Image(img.dataobj, affine)
plotting.plot_stat_map(stat_img, 
                       cut_coords = (-35,-52,-14),
                       title = "GPA",
                       output_file = "C:/Users/Angela Andreella/Documents/Thesis_Doc/paper_priorGPA/Plot/sub7_gpa.pdf")


for i in range(10):
    # img is now an in-memory 3D img
    img = map2nifti(ds_all[i], Xest[i][23,:])
    affine = ds_all[i].a['imgaffine'].value
    stat_img = nib.Nifti1Image(img.dataobj, affine)
    plotting.plot_stat_map(stat_img, 
                           colorbar=True,
                           cut_coords = (-35,-52,-14),
                           #vmax = 1.2,
                           title = "GPA",
                           output_file = "C:/Users/Angela Andreella/Documents/Thesis_Doc/Paper_GPA_Neuro - Copy/sub" + str(i) + "_gpa.pdf")

f = plotting.plot_glass_brain(None)
f.add_contours(stat_img, filled = True)
f.title("GPA")

###########################################GPA0##################################################

data = np.load('C:/Users/Angela Andreella/Documents/Thesis_Doc/Hyperaligment/Computation/ObjectAnalysis/Output/al_gpa0_VT.npz')
data.files
Xest = data['Xest']

img = map2nifti(ds_all[6], Xest[6][11,:])
affine = ds_all[6].a['imgaffine'].value
stat_img = nib.Nifti1Image(img.dataobj, affine)
plotting.plot_stat_map(stat_img, 
                       cut_coords = (-35,-52,-14),
                       title = "von Mises-Fisher-Procrustes model",
                       output_file = "C:/Users/Angela Andreella/Documents/Thesis_Doc/paper_priorGPA/Plot/sub7_gpaPrior.pdf")


for i in range(10):
    # img is now an in-memory 3D img
    img = map2nifti(ds_all[i], Xest[i][23,:])
    affine = ds_all[i].a['imgaffine'].value
    stat_img = nib.Nifti1Image(img.dataobj, affine)
    plotting.plot_stat_map(stat_img, 
                           colorbar=True,
                           cut_coords = (-40,-52,-14),
                           #vmax = 1.2,
                           title = "GPA with prior alignment",
                           output_file = "C:/Users/Angela Andreella/Documents/Thesis_Doc/Paper_GPA_Neuro - Copy/sub" + str(i) + "_gpaPrior.pdf")


f = plotting.plot_glass_brain(None)
f.add_contours(stat_img, filled = True)
f.title("GPA with prior")


