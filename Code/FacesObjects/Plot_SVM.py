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


ds_all = mvpa2.suite.h5load('C:/Users/Angela Andreella/Documents/GitHub/priorGPA/hyperalignment_tutorial_data_2.4.hdf5.gz')

###Hyperalignment

data = np.load('C:/Users/Angela Andreella/Documents/Thesis_Doc/Hyperaligment/Computation/ObjectAnalysis/Output/SVM_coef_hyp.npz')
data = np.load('C:/Users/Angela Andreella/Desktop/New folder1/SVM_coef_hyp.npz')
data = np.load('C:/Users/Angela Andreella/Documents/Thesis_Doc/Hyperaligment/Computation/SVM_coef_hyp1.npz')
data.files
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
sens_mean.shape

contrasts[3]

img = map2nifti(ds_all[0], sens_mean[2,:])
affine = ds_all[0].a['imgaffine'].value
stat_img = nib.Nifti1Image(img.dataobj, affine)
plotting.plot_stat_map(stat_img, 
                       #cut_coords = (-35,-52,-13), vmax  = 0.007,
                       #output_file = "C:/Users/Angela Andreella/Documents/Thesis_Doc/Paper_GPA_Neuro - Copy/MalevsChair_hyp.pdf",
                       title = "Hyperalignment")
#Alternative plot

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
                           output_file = "C:/Users/Angela Andreella/Documents/Thesis_Doc/paper_priorGPA/Plot/" + ctr.replace(' ', '') + "_hyp.pdf")



###Anatomical

data = np.load('C:/Users/Angela Andreella/Documents/Thesis_Doc/Hyperaligment/Computation/ObjectAnalysis/Output/SVM_coef_mni.npz')
data = np.load('//dartfs-hpc/rc/home/w/f003vpw/ObjectAnalysis/Output/SVM_coef_mni.npz')
data.files

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

contrasts[3]
np.percentile(sens_mean, [25, 50, 75])

img = map2nifti(ds_all[0], sen_8[4,:])
affine = ds_all[0].a['imgaffine'].value
stat_img = nib.Nifti1Image(img.dataobj, affine)
plotting.plot_stat_map(stat_img, 
                      # threshold  = np.percentile(sens_mean, [20]),
                       #dim   = 3,
                      # output_file = "C:/Users/Angela Andreella/Documents/Thesis_Doc/Paper_GPA_Neuro - Copy/MalevsChair_an.pdf",
                       title = "Anatomical alignment")

#Alternative plot

for i in range(len(sens_mean)):
    # img is now an in-memory 3D img
    img = map2nifti(ds_all[0], sens_mean[i,:])
    affine = ds_all[0].a['imgaffine'].value
    stat_img = nib.Nifti1Image(img.dataobj, affine)
    ctr = str(contrasts[i][0]) + " VS " + str(contrasts[i][1])
    plotting.plot_stat_map(stat_img, 
                           colorbar=True,
                           cut_coords = (-40,-51,-13),
                           #vmax = 0.002,
                           title = "Anatomical alignment",
                           output_file = "C:/Users/Angela Andreella/Documents/Thesis_Doc/paper_priorGPA/Plot/" + ctr.replace(' ', '') + "_an.pdf")


display = plotting.plot_glass_brain(None)
# Here, we project statistical maps with filled=True
display.add_contours(stat_img, filled=True, levels=[-np.inf, -0.01], colors='r')
display.add_contours(stat_img, filled=True, levels=[0.009, np.inf], colors='b')
display.title("Anatomical alignment")

###GPA PRIOR

data = np.load('C:/Users/Angela Andreella/Documents/Thesis_Doc/Hyperaligment/Computation/SVM_coef_gpaE_parallel1.npz')

sen_gpa0 = data['sen_gpaE']

img = map2nifti(ds_all[0], sen_gpa0[12,:])
affine = ds_all[0].a['imgaffine'].value
stat_img = nib.Nifti1Image(img.dataobj, affine)
plotting.plot_stat_map(stat_img, 
                      # cut_coords = (-25,-51,-10),
                      # output_file = "C:/Users/Angela Andreella/Documents/Thesis_Doc/Paper_GPA_Neuro - Copy/MalevsChair_gpaPrior.pdf",
                       title = "von Mises-Fisher-Procrustes model")

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
                           output_file = "C:/Users/Angela Andreella/Documents/Thesis_Doc/paper_priorGPA/Plot/" + ctr.replace(' ', '') + "_FP.pdf")

data = np.load('C:/Users/Angela Andreella/Desktop/ObjectAnalysis/Output/SVM_coef_gpaE.npz')
data.files
data['bsc_mean']
###################################################GPA #####################################################

data = np.load('C:/Users/Angela Andreella/Desktop/ObjectAnalysis/SVM_coef_gpa0.npz')
data = np.load('//dartfs-hpc/rc/home/w/f003vpw/ObjectAnalysis/Output/SVM_coef_gpa0.npz')
data.files
sen_gpa0 = data['sen_gpa0']


img = map2nifti(ds_all[0], sen_gpa0[3,:])
affine = ds_all[0].a['imgaffine'].value
stat_img = nib.Nifti1Image(img.dataobj, affine)
plotting.plot_stat_map(stat_img, 
                    #   output_file = "C:/Users/Angela Andreella/Documents/Thesis_Doc/Paper_GPA_Neuro - Copy/MalevsChair_gpa.pdf",
                       title = "GPA alignment")


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
                           output_file = "C:/Users/Angela Andreella/Documents/Thesis_Doc/paper_priorGPA/Plot/" + ctr.replace(' ', '') + "_gpa.pdf")

display = plotting.plot_glass_brain(None)
display.add_contours(stat_img, filled=True, levels=[3.], colors='r')
display.title('Same demonstration but using fillings inside contours')

