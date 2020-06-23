"""
Raiders Data creation
"""

#packages
import numpy as np
import nibabel as nib
import mvpa2
from mvpa2.suite import *
from scipy.sparse.linalg import svds
import matplotlib.pyplot as plt
import itertools
import os
npyfilespath ="/MovieData/raiders/data/preprocessed/raiders-8ch"   
os.chdir(npyfilespath)
import glob
from nibabel.freesurfer.io import read_geometry
from mvpa2.support.nibabel.surf import Surface

#You need to download the data from https://github.com/HaxbyLab/raiders_data for EV, LO and VT.

#Load data, 10 subject, 8 runs, left and right hemispher. We use the first scan 8ch with resolution 10242=32**2 *10 + 2
npyfilespath ="/MovieData/raiders/data/preprocessed/raiders-8ch/"   
os.chdir(npyfilespath)

#mask
vt_lh = nib.load('/MovieData/raiders/masks/lh.mask_VT.gii').darrays[0].data.astype(bool)[:10242]
vt_rh = nib.load('/MovieData/raiders/masks/rh.mask_VT.gii').darrays[0].data.astype(bool)[:10242]

#Function

#Function Left hemispher
def loadDataLeft(ID) :
    npyfilespath ="/MovieData/raiders/data/preprocessed/raiders-8ch/"   
    sub = []
    dim = []
    for npfile in sorted(glob.glob(ID +'_ses1_raiders_*_lh.npy')):
        ds = np.load(os.path.join(npyfilespath, npfile))
        if "run01" not in npfile:
            d = ds[8:, vt_lh]
        else:
            d = ds[:, vt_lh]
        sub.append(d)
        dim.append(d.shape)
    ds_final_l = vstack(sub)
    return ds_final_l, dim

#Function Right hemispher
def loadDataRight(ID) :
    npyfilespath ="/MovieData/raiders/data/preprocessed/raiders-8ch/"   
    sub = []
    dim = []
    for npfile in glob.glob(ID +'_ses1_raiders_*_rh.npy'):
        run = npfile
        ds = np.load(os.path.join(npyfilespath, npfile))
        if "run01" not in run:
            d = ds[8:, vt_rh]
        else:
            d = ds[:, vt_rh]
        sub.append(d)
        dim.append(d.shape)
    ds_final_r = vstack(sub)
    return ds_final_r, dim

#Load all subjects and create a list.
subj = ['rid000005','rid000011', 'rid000014', 'rid000015', 'rid000020', 'rid000028', 
        'rid000029', 'rid000033', 'rid000038', 'rid000042', 'rid000043']    

ds_all = list()
shapeL = list()
for i in subj:
    dl, sl = loadDataLeft(ID = i)
    dr, sr = loadDataRight(ID = i)
    ds_final = np.concatenate((dl,dr),axis=1)
    ds_all.append(ds_final)
    shapeL.append(np.array(sl))


f_myfile = open('movie.data', 'wb')
pickle.dump(ds_all, f_myfile)
f_myfile.close()


#Load coordinates

#mask_l = np.load('//dartfs-hpc/rc/home/w/f003vpw/MovieData/searchlight/fsaverage_lh_mask.npy')
#mask_r = np.load('//dartfs-hpc/rc/home/w/f003vpw/MovieData/searchlight/fsaverage_rh_mask.npy')

vertices_l, faces_l = read_geometry('/MovieData/searchlight/fsaverage/surf/lh.midthickness')
vertices_r, faces_r = read_geometry('/MovieData/searchlight/fsaverage/surf/rh.midthickness')

vertices_l[:10242]
vertices_r[:10242]

surf_l = Surface(vertices_l, faces_l)
surf_r = Surface(vertices_r, faces_r)

vt_idx_l = np.where(vt_lh)[0]
vt_idx_r = np.where(vt_rh)[0]

coord_l = surf_l.vertices[vt_idx_l,:]
coord_r = surf_l.vertices[vt_idx_r,:]

coord = np.concatenate((coord_l,coord_r),axis=0)


np.savez('/Data/Raiders/Raiders_data/coord.npz', 
         surf_l = surf_l, surf_r = surf_r, coord = coord)

#For each ROI of interest
