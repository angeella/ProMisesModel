# The von Mises-Fisher-Procrustes model
[![DOI](https://zenodo.org/badge/224435643.svg)](https://zenodo.org/badge/latestdoi/224435643)

The method is implemented in Python according to the [PyMVPA](http://www.pymvpa.org/index.html) (MultiVariate Pattern Analysis in Python) package. 

First of all, the **The von Mises-Fisher-Procrustes model** could be describe as the Generalized Procrustes Analysis with prior information, i.e. spatial anatomical brain information in the fMRI framework, imposed on the orthogonal matrix parameter. The von Mises-Fisher-Procrustes model allows to improve the between-subject analysis in fMRI data, considering both anatomical and functional characteristics as information in the estimation process. It is a useful functional alignment to perfom for between-subjects analysis. It must be used after data preprocessing, as motion correction, anatomical alignment and so on. The method is computationally heavy, at the moment, for that we recommend to use a mask from FSL, Matlab or other software. The mask must be used without falling into the double dipping problem. You must choose the mask before seeing the data, having some prior knowledge about the correlation between the region of interest and the type of task-design activation.

We are working also on a new method that permits to align the whole brain, so.. stay tuned! :)

Let's start with this short tutorial to understand easily how apply this functional alignment.

First of all, you need to install Python, and the PyMVPA, decimal, math, multiprocessing, nibabel, scipy, pydicom, nilearn, dipy, pickle packages. If it is your first time with Python, you can see my [repository](https://github.com/angeella/Python_Tutorial) where I explain how install packages, how manage Anaconda and so on.

So, you need to download the script [vMFPmodel.py](https://github.com/angeella/vMFPmodel/blob/master/vMFPmodel.py) and put on the folder where you are working. You must name the script [vMFPmodel.py](https://github.com/angeella/vMFPmodel/blob/master/vMFPmodel.py) as **vMFPmodel.py**.

After the installation of the packages in Anaconda, you need to import the packages:

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
os.chdir('put_your_path_where_vMFPmodel.py_is')
```
import the function **vMFPmodel.py**:

```python
from vMFPmodel import vMFPmodel
```
So, we need to import our data. The data are composed by nii images (one for each subject), preprocessed by FSL, or Matlab, etc. You need to rename your data as sub-x, where x goes from 01 to the last index of your subject that we will call ```idxmax```. Also, you need to set your ```path```, where the vMFPmodel.py file is. Also, in the same folder where your data are and where the vMFPmodel.py file is, you must put the mask named ```mask``` as nifti files. If you want to learn how produce a mask using FSL, please refers to this simple [tutorial](https://www.youtube.com/watch?v=fIu4tUjRfUE).

So, you need to modify the first two lines of this first part of code:

```python
path = "path_where_your_data_are"   #Here you set your path, finish it with /
idxmax = #Here you must put the last numeric index of your set of subjects
idx = np.hstack([ '0'+ str(x) for x in range(1,idxmax)]) 

img_dt = []
for ds in range(len(idx)):
    img_o = nib.load(path + str(idx[ds]) + '.nii.gz')
    mask_img = nib.load(path + 'mask.nii.gz')
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
Then, we perform some preprocessing steps:

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

Then, we apply the von Mises-Fisher-Procrustes model, using as prior information the three-dimensional coordinates of the voxels. At first, we create the similarity euclidean matrix called $Q$:

```python
coord = img[0].fa.voxel_indices
dist = distance.cdist(np.array(coord), np.array(coord), "euclidean")
Q = np.exp(-dist)
```
Then, we need to tune the concentration hyperparameter $k$ of the Fisher Von Mises prior distribution:

```python
logSum = []
nOO = []

for k in range(len(kval)):
    for ns in range(nsubjs):            
        nLogic = [np.unique(sd[0].sa['subject'].value)[0]!=ns for sd in img]
        for i in range(nsubjs):
            if nLogic[i]:
                nOO.append(i)
        ds_LOO = [img_Right[i] for i in nOO]
        hyper = vMFPmodel(maxIt = 10, t = 0.01, k = k, Q = Q, ref_ds = None,  scaling=True, reflection = True, subj=False)
        hypmaps = hyper.gpa(datasets=ds_LOO)
        R = hypmaps[2]
        Xest = hypmaps[0]
        M = hypmaps[4]
        log = -np.sum([np.linalg.norm(x- M, ord = 'fro') for x in Xest])
        
    logSum.append(log)    

logSumK = zip(kval,logSum)
khat = min(logSumK, key = lambda t: t[1])[0]
 
hyper = vMFPmodel(maxIt = 10, t = 0.001, k = khat, Q = Q, ref_ds = None,  scaling=True, reflection = True, subj=False)

hypmaps = hyper.gpa(datasets=img)

Xest = hypmaps[0]
```

So, we have our aligned brain images called ```Xest```! Let's save it as nii file:

```python
out = img

#be sure that we work in float
for ds in range(len(out)): 
    out[ds].samples = np.column_stack((Xest[ds].astype(np.float64), Xest[ds].astype(np.float64)))
    
for ds in range(len(out)):
  nimg = mvpa2.datasets.mri.map2nifti(out[ds])
  nib.save(nimg, path + 'sub-' + str(idx[ds]) + 'vMFPmodel_align.nii.gz')
```
That's it! You can use these aligned brain images in FSL, Matlab etc to perform between subject analysis :)

If you want to compare the results using the [Hyperalignment](https://www.sciencedirect.com/science/article/pii/S0896627311007811?via%3Dihub) method, you can check the [tutorial](http://www.pymvpa.org/examples/hyperalignment.html) from the PyMVPA package.


# Did you find some bugs?

Please write to angela.andreella[\at]stat[\dot]unipd[\dot]it or insert a reproducible example using [reprex](https://github.com/tidyverse/reprex) on my [issue github page](https://github.com/angeella/vMFPmodel/issues).

