################################################################################
##############################SVM COEFFICIENT PLOT##############################
################################################################################


import numpy as np
from mvpa2.datasets.mri import map2nifti
import mvpa2
from mvpa2.suite import * 
from nilearn import plotting

contrasts = [('DogFace', 'Chair'), ('FemaleFace', 'Chair'), ('House', 'Chair'),
       ('MaleFace', 'Chair'), ('MonkeyFace', 'Chair'), ('Shoe', 'Chair'),
       ('FemaleFace', 'DogFace'), ('House', 'DogFace'),
       ('MaleFace', 'DogFace'), ('MonkeyFace', 'DogFace'),
       ('Shoe', 'DogFace'), ('House', 'FemaleFace'),
       ('MaleFace', 'FemaleFace'), ('MonkeyFace', 'FemaleFace'),
       ('Shoe', 'FemaleFace'), ('MaleFace', 'House'),
       ('MonkeyFace', 'House'), ('Shoe', 'House'),
       ('MonkeyFace', 'MaleFace'), ('Shoe', 'MaleFace'),
       ('Shoe', 'MonkeyFace')]


ds_all = hd.h5load("C:/Users/Angela Andreella/Documents/GitHub/ProMisesModel/Data/Faces_Objects/hyperalignment_tutorial_data_2.4.hdf5.gz")

###Hyperalignment

data = np.load('SVM_coef_hyp.npz')

sen_1_mean_perm = data['sen_1_mean_perm']
sen_2_mean_perm = data['sen_2_mean_perm']
sen_3_mean_perm = data['sen_3_mean_perm']
sen_4_mean_perm = data['sen_4_mean_perm']
sen_5_mean_perm = data['sen_5_mean_perm']
sen_6_mean_perm = data['sen_6_mean_perm']
sen_7_mean_perm = data['sen_7_mean_perm']
sen_8_mean_perm = data['sen_8_mean_perm']
sen_9_mean_perm = data['sen_9_mean_perm']
sen_10_mean_perm = data['sen_10_mean_perm']

s1 = np.mean([sen_1_mean_perm[i,:,:] for i in range(len(sen_1_mean_perm))], axis = 0)
s2 = np.mean([sen_2_mean_perm[i,:,:] for i in range(len(sen_2_mean_perm))], axis = 0)
s3 = np.mean([sen_3_mean_perm[i,:,:] for i in range(len(sen_3_mean_perm))], axis = 0)
s4 = np.mean([sen_4_mean_perm[i,:,:] for i in range(len(sen_4_mean_perm))], axis = 0)
s5 = np.mean([sen_5_mean_perm[i,:,:] for i in range(len(sen_5_mean_perm))], axis = 0)
s6 = np.mean([sen_6_mean_perm[i,:,:] for i in range(len(sen_6_mean_perm))], axis = 0)
s7 = np.mean([sen_7_mean_perm[i,:,:] for i in range(len(sen_7_mean_perm))], axis = 0)
s8 = np.mean([sen_8_mean_perm[i,:,:] for i in range(len(sen_8_mean_perm))], axis = 0)
s9 = np.mean([sen_9_mean_perm[i,:,:] for i in range(len(sen_9_mean_perm))], axis = 0)
s10 = np.mean([sen_10_mean_perm[i,:,:] for i in range(len(sen_10_mean_perm))], axis = 0)

sens_mean = np.mean([s1,s2,s3,s4,s5,s6,s7,s8,s9,s10],axis=0)

for i in range(len(sens_mean)):
    # img is now an in-memory 3D img
    img = map2nifti(ds_all[0], sens_mean[i,:])
    affine = ds_all[0].a['imgaffine'].value
    stat_img = nib.Nifti1Image(img.dataobj, affine)
    ctr = str(contrasts[i][0]) + " VS " + str(contrasts[i][1])
    plotting.plot_stat_map(stat_img, 
                           cut_coords = (-40,-51,-13),
                           #vmax = 0.002,
                           colorbar=True,
                           title = "Hyperalignment",
                           output_file = ctr.replace(' ', '') + "_hyp.pdf")

###Anatomical

data = np.load('SVM_coef_mni.npz')

sen_1 = data['sen_1']
sen_2 = data['sen_2']
sen_3 = data['sen_3']
sen_4 = data['sen_4']
sen_5 = data['sen_5']
sen_6 = data['sen_6']
sen_7 = data['sen_7']
sen_8 = data['sen_8']
sen_9 = data['sen_9']
sen_10 = data['sen_10']

sens_mean = np.mean([sen_1,sen_2,sen_3,
                     sen_4,sen_5,sen_6,
                     sen_7,sen_8,sen_9,sen_10],axis=0)

for i in range(len(sens_mean)):
    # img is now an in-memory 3D img
    img = map2nifti(ds_all[0], sens_mean[i,:])
    affine = ds_all[0].a['imgaffine'].value
    stat_img = nib.Nifti1Image(img.dataobj, affine)
    ctr = str(contrasts[i][0]) + " VS " + str(contrasts[i][1])
    plotting.plot_stat_map(stat_img, 
                           colorbar=True,
                           cut_coords = (-40,-51,-13),
                           title = "Anatomical alignment",
                           output_file = ctr.replace(' ', '') + "_an.pdf")


###GPA PRIOR

data = np.load('SVM_coef_gpaE.npz')

sen_gpa0 = data['sen_gpaE']

#ALTERNATIVE PLOT
for i in range(len(sen_gpa0)):
    # img is now an in-memory 3D img
    img = map2nifti(ds_all[0], sen_gpa0[i,:])
    affine = ds_all[0].a['imgaffine'].value
    stat_img = nib.Nifti1Image(img.dataobj, affine)
    ctr = str(contrasts[i][0]) + " VS " + str(contrasts[i][1])
    plotting.plot_stat_map(stat_img, 
                           colorbar=True,
                           cut_coords = (-40,-51,-13),
                          # vmax = 0.002,
                           title = "von Mises-Fisher-Procrustes model",
                           output_file = ctr.replace(' ', '') + "_FP.pdf")


###################################################GPA #####################################################

data = np.load('SVM_coef_gpa0.npz')

sen_gpa0 = data['sen_gpa0']

#ALTERNATIVE PLOT
for i in range(len(sen_gpa0)):
    # img is now an in-memory 3D img
    img = map2nifti(ds_all[0], sen_gpa0[i,:])
    affine = ds_all[0].a['imgaffine'].value
    stat_img = nib.Nifti1Image(img.dataobj, affine)
    ctr = str(contrasts[i][0]) + " VS " + str(contrasts[i][1])
    plotting.plot_stat_map(stat_img, 
                           cut_coords = (-40,-51,-13),
                           #vmax = 0.002,
                           colorbar=True,
                           title = "GPA alignment",
                           output_file = ctr.replace(' ', '') + "_gpa.pdf")



