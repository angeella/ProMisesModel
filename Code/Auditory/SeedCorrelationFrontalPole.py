# -*- coding: utf-8 -*-
"""
Seed-to-voxel correlation
"""

#Load packages 

from nilearn import input_data
import numpy as np
import nibabel as nib
from nilearn import plotting
import matplotlib.pyplot as plt

#################Time series extraction#####################

#Superior Temporal Gyrus
fp_coords = [( 0, 64, 18)]

seed_masker = input_data.NiftiSpheresMasker(
    fp_coords, radius=8,
    detrend=True, standardize=True,
    low_pass=0.1, high_pass=0.01, t_r=2,
    memory='nilearn_cache', memory_level=1, verbose=0)

#Load data without functional alignment and save it in img_dt as list like:

path = "Anatomical"
idx = np.hstack(['021',[ '0'+ str(x) for x in range(23,32)], [ '0'+ str(x) for x in range(33,41)]])

img_dt = []
for ds in range(len(idx)):
    img_o = nib.load(path + '/sub-' + str(idx[ds]) + '.feat.nii.gz')
    img_dt.append(img_o.get_data())

mean_subj = np.mean(img_dt, axis = 0)
affine = img_o.get_affine()
mean_subj_img = nib.Nifti1Image(mean_subj, affine =[[  -2.,    0.,    0.,   90.],
                                           [   0.,    2.,    0., -126.],
                                           [   0.,    0.,    2.,  -72.],
                                           [   0.,    0.,    0.,    1.]])

seed_time_series = seed_masker.fit_transform(mean_subj_img)

brain_masker = input_data.NiftiMasker(
    smoothing_fwhm=6,
    detrend=True, standardize=True,
    low_pass=0.1, high_pass=0.01, t_r=2,
    memory='nilearn_cache', memory_level=1, verbose=0)

brain_time_series = brain_masker.fit_transform(mean_subj_img)

seed_to_voxel_correlations = (np.dot(brain_time_series.T, seed_time_series) /
                              seed_time_series.shape[0]
                              )

seed_to_voxel_correlations_img = brain_masker.inverse_transform(
    seed_to_voxel_correlations.T)

display = plotting.plot_stat_map(seed_to_voxel_correlations_img,
                                # threshold=0.5, 
                                 vmax=1,
                                 cut_coords=fp_coords[0],
                                 title="Anatomical alignment"
                                 )
display.add_markers(marker_coords=fp_coords, marker_color='black',
                    marker_size=300)

#Load data aligned by ProMises model

mean_img = np.load("mean_ProMises.npz")['arr_0']

mean_subj = np.reshape(np.transpose(mean_img), (91, 109, 91, 310))
mean_subj_img = nib.Nifti1Image(mean_subj, affine = affine)

seed_time_series = seed_masker.fit_transform(mean_subj_img)

brain_masker = input_data.NiftiMasker(
    smoothing_fwhm=6,
    detrend=True, standardize=True,
    low_pass=0.1, high_pass=0.01, t_r=2,
    memory='nilearn_cache', memory_level=1, verbose=0)

brain_time_series = brain_masker.fit_transform(mean_subj_img)

seed_to_voxel_correlations = (np.dot(brain_time_series.T, seed_time_series) /
                              seed_time_series.shape[0]
                              )

seed_to_voxel_correlations_img = brain_masker.inverse_transform(
    seed_to_voxel_correlations.T)

display = plotting.plot_stat_map(seed_to_voxel_correlations_img,
                                # threshold=0.3, 
                                 vmax=1,
                                 cut_coords=fp_coords[0],
                                 title="ProMises model"
                                 )
display.add_markers(marker_coords=fp_coords, marker_color='black',
                    marker_size=300)



