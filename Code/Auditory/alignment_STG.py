import numpy as np
#import nibabel
#import scipy
#from scipy.sparse.linalg import svds
#import itertools 
import mvpa2
from mvpa2.suite import *
#if __debug__:
#    from mvpa2.base import debug
import os
os.chdir('//dartfs-hpc/rc/home/w/f003vpw/AuditoryData') #your path
from priorGPA_parallel import priorHyA
#import pickle
#from function import distance_pairwise
import nibabel as nib
from scipy.spatial import distance
from scipy.ndimage import zoom
#import pydicom
from nilearn.masking import apply_mask
import nilearn
from dipy.align import reslice
#from dipy.data import get_fnames
import dipy
import nilearn
from nilearn.input_data import NiftiMasker
from dipy.align.reslice import reslice
import mvpa2.base.hdf5 as hd

path = "//dartfs-hpc/rc/home/w/f003vpw/AuditoryData/Data/"
idx = np.hstack(['021',[ '0'+ str(x) for x in range(23,32)], [ '0'+ str(x) for x in range(33,41)]])

#img_dt = []
#for ds in range(len(idx)):
#  img_o = nib.load(path + 'sub-' + str(idx[ds]) + 'masked.nii.gz')
#  mask_img = nib.load(path + 'mask_Superior_Temporal_Gyrus.nii.gz')
  #affine = mask_img.affine
#  affine = img_o.affine
#  img_o.header.get_zooms()
#  mask_img.header.get_zooms()
#  new_zooms = (2., 2., 2.)
#  mask, mask_affine = dipy.align.reslice.reslice(mask_img.get_data(), mask_img.affine, mask_img.header.get_zooms(), new_zooms)
  #mask_affine
  #img_o.affine
#  mask = nib.Nifti1Image(mask, mask_affine)
#  img_dt.append(mvpa2.datasets.mri.fmri_dataset(img_o, mask = mask))

img = []
for ds in range(len(idx)):
#for ds in range(3):
#    img_o = nib.load(path + 'sub-' + str(idx[ds]) + 'masked.nii.gz')
#    data = img_o.get_data()
#    affine = img_o.affine
#    zooms = img_o.header.get_zooms()[:3]
#    new_zooms = (12., 12., 12.)
#    data2, affine2 = reslice(data, affine, zooms, new_zooms)
#    img2 = nib.nifti1.Nifti1Image(data2, affine2)
#    img.append(mvpa2.datasets.mri.fmri_dataset(path + 'sub-' + str(idx[ds]) + 'masked_gyrus.nii.gz')) 
     img.append(hd.h5load(path + 'sub-'+ str(idx[ds]) + 'STG.hdf5'))  


# inject the subject ID into all datasets
for i, sd in enumerate(img):
    sd.sa['subject'] = np.repeat(i, len(sd))
    
#be sure that we work in float
for ds in range(len(img)):
    img[ds].samples = img[ds].samples.astype(np.float64)   
  
# number of subjects
nsubjs = len(img)

#Divide the dataset in left and right

coord = []
img_Right = []
for ds in range(len(idx)):
    img_Right.append(Dataset(img[ds].samples[:,:4560]))
    coord.append(img[ds].fa["voxel_indices"][:4560,])
    img_Right[ds].fa.voxel_indices = coord[ds]
    img_Right[ds].sa['time_coords'] = img[ds].sa['time_coords']
    img_Right[ds].sa['subject'] = np.repeat(ds, 310)
    img_Right[ds].sa['time_indices'] = img[ds].sa['time_indices']
#    img_Right[ds].a['mapper'] = img[ds].a['mapper']
#    img_Right[ds].a['imgaffine'] = img[ds].a['imgaffine']
#    img_Right[ds].a['voxel_eldim'] = img[ds].a['voxel_eldim']
#   img_Right[ds].a['imgtype'] = img[ds].a['imgtype']
#    img_Right[ds].a['imghdr'] = img[ds].a['imghdr']
#    img_Right[ds].a['imgtype'] = img[ds].a['imgtype']
#    img_Right[ds].a['voxel_dim'] = img[ds].a['voxel_dim']
    
coord = []
img_Left = []
for ds in range(len(idx)):
    img_Left.append(Dataset(img[ds].samples[:,4560:]))
    coord.append(img[ds].fa["voxel_indices"][4560:,])
    img_Left[ds].fa.voxel_indices = coord[ds]
    img_Left[ds].sa['time_coords'] = img[ds].sa['time_coords']
    img_Left[ds].sa['subject'] = np.repeat(ds, 310)
    img_Left[ds].sa['time_indices'] = img[ds].sa['time_indices']
#    img_Left[ds].a['mapper'] = img[ds].a['mapper']
#   img_Left[ds].a['imgaffine'] = img[ds].a['imgaffine']
#    img_Left[ds].a['voxel_eldim'] = img[ds].a['voxel_eldim']
#    img_Left[ds].a['imgtype'] = img[ds].a['imgtype']
#   img_Left[ds].a['imghdr'] = img[ds].a['imghdr']
#    img_Left[ds].a['imgtype'] = img[ds].a['imgtype']
#   img_Left[ds].a['voxel_dim'] = img[ds].a['voxel_dim']

#####################################################################################################
#######################################Hyperalignment#######################################
#####################################################################################################

hyper = Hyperalignment()
hypmaps = hyper(img_Right)

R = [d.proj for d in hypmaps]
X = [h.samples for h in img_Right]
Xest_Right = [np.dot(x,r) for x,r in zip(X,R)]

hyper = Hyperalignment()
hypmaps = hyper(img_Left)

R = [d.proj for d in hypmaps]
X = [h.samples for h in img_Left]
Xest_Left = [np.dot(x,r) for x,r in zip(X,R)]

out = img

#be sure that we work in float
for ds in range(len(out)):
    out[ds].samples = np.column_stack((Xest_Right[ds].astype(np.float64), Xest_Left[ds].astype(np.float64)))

for ds in range(len(out)):
  nimg = mvpa2.datasets.mri.map2nifti(out[ds])
#  data = nimg.get_data()
#  affine = nimg.affine
 # zooms = nimg.header.get_zooms()[:3]
#  new_zooms = (2., 2., 2.)
# data2, affine2 = reslice(data, affine, zooms, new_zooms)
#  img2 = nib.nifti1.Nifti1Image(data2, affine2)
  nib.save(nimg, path + 'sub-' + str(idx[ds]) + 'STG_Hyper.nii.gz')

#####################################################################################################
#######################################GPA##############################################################################
#####################################################################################################

hyper = priorHyA(maxIt = 10, t = 0.001, k = 0, Q = None, ref_ds = None,  scaling=True, reflection = True, subj=False)
    
hypmaps = hyper.gpa(datasets=img_Right)

R = hypmaps[2]
X = [h.samples for h in img_Right]
Xest_Right = [np.dot(x,r) for x,r in zip(X,R)]

hyper = priorHyA(maxIt = 10, t = 0.001, k = 0, Q = None, ref_ds = None,  scaling=True, reflection = True, subj=False)

hypmaps = hyper.gpa(datasets=img_Left)

R = hypmaps[2]
X = [h.samples for h in img_Left]
Xest_Left = [np.dot(x,r) for x,r in zip(X,R)]

out = img

#be sure that we work in float
for ds in range(len(out)): 
    out[ds].samples = np.column_stack((Xest_Right[ds].astype(np.float64), Xest_Left[ds].astype(np.float64)))
    
for ds in range(len(out)):
  nimg = mvpa2.datasets.mri.map2nifti(out[ds])
#  data = nimg.get_data()
#  affine = nimg.affine
#  zooms = nimg.header.get_zooms()[:3]
#  new_zooms = (2., 2., 2.)
#  data2, affine2 = reslice(data, affine, zooms, new_zooms)
#  img2 = nib.nifti1.Nifti1Image(data2, affine2)
  nib.save(nimg, path + 'sub-' + str(idx[ds]) + 'STG_GPA.nii.gz')

#####################################################################################################
###############################################GPA PRIOR###############################################
#####################################################################################################

kval =  np.arange(101)

coord = img_Right[0].fa.voxel_indices
dist = distance.cdist(np.array(coord), np.array(coord), "euclidean")
Q = np.exp(-dist/coord.shape[0])
logSum = []
nOO = []

for k in range(len(kval)):
    for ns in range(nsubjs):            
        nLogic = [np.unique(sd[0].sa['subject'].value)[0]!=ns for sd in img_Right]
        for i in range(nsubjs):
            if nLogic[i]:
                nOO.append(i)
        ds_LOO = [img_Right[i] for i in nOO]
        hyper = priorHyA(maxIt = 10, t = 0.01, k = k, Q = Q, ref_ds = None,  scaling=True, reflection = True, subj=False)
        hypmaps = hyper.gpa(datasets=ds_LOO)
        R = hypmaps[2]
        Xest = hypmaps[0]
        M = hypmaps[4]
        log = -np.sum([np.linalg.norm(x- M, ord = 'fro') for x in Xest])
        
    logSum.append(log)    

logSumK = zip(kval,logSum)
khat = min(logSumK, key = lambda t: t[1])[0]


hyper = priorHyA(maxIt = 10, t = 0.01, k = khat, Q = Q, ref_ds = None,  scaling=True, reflection = True, subj=False)
hypmaps = hyper.gpa(datasets=img_Right)        
Xest_Right = [h.forward(sd) for h, sd in zip(hypmaps[1], img_Right)]
distHprior = distance_pairwise(Xest_Right)
traceHprior = traceProcrustes(Xest_Right)
RGPAprior=hypmaps[2]
distItprior=hypmaps[3]

coord = img_Left[0].fa.voxel_indices
dist = distance.cdist(np.array(coord), np.array(coord), "euclidean")
Q = np.exp(-dist/coord.shape[0])
logSum = []
nOO = []

for k in range(len(kval)):
    for ns in range(nsubjs):            
        nLogic = [np.unique(sd[0].sa['subject'].value)[0]!=ns for sd in img_Left]
        for i in range(nsubjs):
            if nLogic[i]:
                nOO.append(i)
        ds_LOO = [img_Right[i] for i in nOO]
        hyper = priorHyA(maxIt = 10, t = 0.01, k = k, Q = Q, ref_ds = None,  scaling=True, reflection = True, subj=False)
        hypmaps = hyper.gpa(datasets=ds_LOO)
        R = hypmaps[2]
        Xest = hypmaps[0]
        M = hypmaps[4]
        log = -np.sum([np.linalg.norm(x- M, ord = 'fro') for x in Xest])
        
    logSum.append(log)    

logSumK = zip(kval,logSum)
khat = min(logSumK, key = lambda t: t[1])[0]

hyper = priorHyA(maxIt = 10, t = 0.01, k = khat, Q = Q, ref_ds = None,  scaling=True, reflection = True, subj=False)
hypmaps = hyper.gpa(datasets=img_Left)        
Xest_Left = [h.forward(sd) for h, sd in zip(hypmaps[1], img_Left)]
distHprior = distance_pairwise(Xest_Left)
traceHprior = traceProcrustes(Xest_Left)
RGPAprior=hypmaps[2]
distItprior=hypmaps[3]

out = img

#be sure that we work in float
for ds in range(len(out)): 
    out[ds].samples = np.column_stack((Xest_Right[ds].astype(np.float64), Xest_Left[ds].astype(np.float64)))
    
for ds in range(len(out)):
  nimg = mvpa2.datasets.mri.map2nifti(out[ds])
#  data = nimg.get_data()
#  affine = nimg.affine
#  zooms = nimg.header.get_zooms()[:3]
#  new_zooms = (2., 2., 2.)
#  data2, affine2 = reslice(data, affine, zooms, new_zooms)
#  img2 = nib.nifti1.Nifti1Image(data2, affine2)
  nib.save(nimg, path + 'sub-' + str(idx[ds]) + 'STG_vMFP.nii.gz')
