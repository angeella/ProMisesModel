# -*- coding: utf-8 -*-
"""
TEST
"""

import numpy as np
import mvpa2
from mvpa2.suite import *
import os
os.chdir('C:/Users/Angela Andreella/Documents/Thesis_Doc/Hyperaligment/Computation/Test')
from priorGPA import priorGPA
#Load dataset
ds_all = h5load('//dartfs-hpc/rc/home/w/f003vpw/ObjectAnalysis/data/hyperalignment_tutorial_data_2.4.hdf5.gz')
#Number of voxels to select
nf = 80
# Automatic feature selection
fselector = FixedNElementTailSelector(nf, tail='upper',mode='select', sort=False)
anova = OneWayAnova()
fscores = [anova(sd) for sd in ds_all]
featsels = [StaticFeatureSelection(fselector(fscore)) for fscore in fscores]
ds_all_fs = [fs.forward(sd) for fs, sd in zip(featsels, ds_all)]
#Compute Location Parameter as Similarity Matrix using the euclidean distance of the three dimensional coordinates of the voxels
coord = [ds.fa.voxel_indices for ds in ds_all_fs]
dist = [cdist(np.array(c), np.array(c), "euclidean") for c in coord]
Q = [np.exp(-d/c.shape[0]) for d,c in zip(dist,coord)]
#Alignment step
gp = priorGPA(maxIt = 10, t = 0.001, k = 1, Q = Q, ref_ds = None,  scaling=True, reflection = True, subj=True)
mappers = gp.gpa(ds_all_fs)[1]
len(mappers)








