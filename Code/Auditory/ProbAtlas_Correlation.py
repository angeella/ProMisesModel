# -*- coding: utf-8 -*-
"""
Extracting signals of a probabilistic atlas of functional regions
"""

#ATLAS
from nilearn import datasets
from nilearn.input_data import NiftiMapsMasker
import numpy as np
import nibabel as nib
from nilearn import plotting
from nilearn.connectome import ConnectivityMeasure
import numpy as np
import matplotlib.pyplot as plt

atlas = datasets.fetch_atlas_msdl()
# Loading atlas image stored in 'maps'
atlas_filename = atlas['maps']
# Loading atlas data stored in 'labels'
labels = atlas['labels']

#Extract time series

masker = NiftiMapsMasker(maps_img=atlas_filename, standardize=True,
                         memory='nilearn_cache', verbose=5)


#Load data aligned by ProMises model

mean_img = np.load("mean_ProMises.npz")['arr_0']

mean_subj = np.reshape(np.transpose(mean_img), (91, 109, 91, 310))
mean_subj_img = nib.Nifti1Image(mean_subj, affine = [[  -2.,    0.,    0.,   90.],
                                           [   0.,    2.,    0., -126.],
                                           [   0.,    0.,    2.,  -72.],
                                           [   0.,    0.,    0.,    1.]])

time_series = masker.fit_transform(mean_subj_img)

correlation_measure = ConnectivityMeasure(kind='correlation')
correlation_matrix = correlation_measure.fit_transform([time_series])[0]

# Display the correlation matrix
# Mask out the major diagonal
np.fill_diagonal(correlation_matrix, 0)
display = plotting.plot_matrix(correlation_matrix, labels=labels, colorbar=True,
                     vmax=0.8, vmin=-0.8)
                     
coords = atlas.region_coords

# We threshold to keep only the 20% of edges with the highest value
# because the graph is very dense

display = plotting.plot_connectome(correlation_matrix, coords,
                         edge_threshold="80%", colorbar=True)

#Load data without functional alignment and save it in img_dt as list like:

path = "Anatomical"
idx = np.hstack(['021',[ '0'+ str(x) for x in range(23,32)], [ '0'+ str(x) for x in range(33,41)]])

img_dt = []
for ds in range(len(idx)):
    img_o = nib.load(path + '/sub-' + str(idx[ds]) + '.feat.nii.gz')
    img_dt.append(img_o.get_data())

mean_subj = np.mean(img_dt, axis = 0)
affine = img_o.get_affine()
mean_subj_img = nib.Nifti1Image(mean_subj, affine = affine)

time_series = masker.fit_transform(mean_subj_img)

correlation_measure = ConnectivityMeasure(kind='correlation')
correlation_matrix1 = correlation_measure.fit_transform([time_series])[0]


# Display the correlation matrix
# Mask out the major diagonal
np.fill_diagonal(correlation_matrix1, 0)
display = plotting.plot_matrix(correlation_matrix1, labels=labels, colorbar=True,
                     vmax=0.8, vmin=-0.8)

coords = atlas.region_coords

# We threshold to keep only the 20% of edges with the highest value
# because the graph is very dense
display = plotting.plot_connectome(correlation_matrix1, coords,
                         edge_threshold="80%", colorbar=True)



