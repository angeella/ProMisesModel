# Generalized Procrustes Analysis with Prior Information
DOI: 10.5281/zenodo.3629269

The method is implemented in Python according to the [PyMVPA](http://www.pymvpa.org/index.html) (MultiVariate Pattern Analysis in Python) package. 

First of all, the Generalized Procrustes Analysis with Prior information, i.e. spatial anatomical brain information, allows to improve the between-subject analysis in fMRI data. It is a useful functional alignment to perfom if someone want to analyze a set of subject. It must be used after data preprocessing, as motion correction, anatomical alignment and so on. The method is computationally heavy, at the moment, for that we recommend to use a mask from FSL, Matlab or other software. The mask must be used without falling into the double dipping problem. So, you must choose the mask after seeing the data, having some prior knowledge about the correlation between the region of interest and the type of task-design activation.

We are working also on a new method that permits to align the whole brain, so.. stay tuned! :)

Let's start with this short tutorial to understand easily how apply this functional alignment.

First of all, you need to install Python, and the PyMVPA, decimal, math, multiprocessing, nibabel, scipy, pydicom, nilearn, dipy, pickle packages. If it is your first time with Python, you can see my [repository](https://github.com/angeella/Python_Tutorial) where I explain how install packages, how manage Anaconda and so on.

So, you need to download the script [priorGPA.py](https://github.com/angeella/priorGPA/blob/master/priorGPA.py) and put on the folder where you are working. 

After the installation, you need to import the packages:
```python
import numpy as np
import mvpa2
from mvpa2.suite import *
import os
import nibabel as nib
from scipy.spatial import distance
from scipy.ndimage import zoom
import pydicom
from nilearn.masking import apply_mask
from dipy.align import reslice
import dipy
import nilearn
from nilearn.input_data import NiftiMasker
import pickle
```
Setup your path:

```python
os.chdir('your_path_where_priorGPA.py_is')
```
The data are composed by nii images (one for each subject), preprocessed by FSL, Matlab and so on. You need to rename your data as sub-x, where x goes from 01 to the last index of your subject that we will call idxmax.

```python
path = "path_where_your_data_are"
idx = np.hstack([ '0'+ str(x) for x in range(1,idxmax)])

img_dt = []
for ds in range(len(idx)):
    img_o = nib.load(path + str(idx[ds]) + '.nii.gz')
    mask_img = nib.load(path + 'your_mask.nii.gz')
    #affine = mask_img.affine
    affine = img_o.affine
    img_o.header.get_zooms()
    mask_img.header.get_zooms()
    new_zooms = (2., 2., 2.)
    mask, mask_affine = dipy.align.reslice.reslice(mask_img.get_data(), mask_img.affine, mask_img.header.get_zooms(), new_zooms)
    #mask_affine
    #img_o.affine
    mask = nib.Nifti1Image(mask, mask_affine)
    img_dt.append(mvpa2.datasets.mri.fmri_dataset(img_o, mask = mask))  
```
Some preprocessing steps:

```python
# inject the subject ID into all datasets
for i, sd in enumerate(img):
    sd.sa['subject'] = np.repeat(i, len(sd))
    
#be sure that we work in float
for ds in range(len(img)):
    img[ds].samples = img[ds].samples.astype(np.float64)   
  
# number of subjects
nsubjs = len(img)
```

```python
coord = img[0].fa.voxel_indices
dist = distance.cdist(np.array(coord), np.array(coord), "euclidean")
Q = np.exp(-dist)
hyper = priorGPA(maxIt = 10, t = 0.001, k = 2, Q = Q, ref_ds = None,  scaling=True, reflection = True, subj=False)

hypmaps = hyper.gpaSub(datasets=img)

Xest = hypmaps[0]
```
So, we have our aligned brain images! Let's save it as nii file:

```python
out = img

#be sure that we work in float
for ds in range(len(out)): 
    out[ds].samples = np.column_stack((Xest[ds].astype(np.float64), Xest[ds].astype(np.float64)))
    
for ds in range(len(out)):
  nimg = mvpa2.datasets.mri.map2nifti(out[ds])
  nib.save(nimg, path + 'sub-' + str(idx[ds]) + 'GPAprior_align.nii.gz')
```
That's it! You can use these aligned brain images in FSL, Matlab etc to perform between subject analysis :)

If you want to compare the results using the [Hyperalignment](https://www.sciencedirect.com/science/article/pii/S0896627311007811?via%3Dihub) method, you can check the [tutorial](http://www.pymvpa.org/examples/hyperalignment.html) from the PyMVPA package.
