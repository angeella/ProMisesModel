#!/usr/bin/env python

"""
Final analysis Object Analysis.
Features selection different for each subjects.
In order to select the best model, the parameter k, we use the Nested Cross Validation method

where d is an indicator of the effective number of parameters in the model.

The dataset is the same use in Haxby 2011, Object visualization recognization in ventral temporal cortex.

"""
import multiprocessing as mp #Parallelizing 
import numpy as np
import nibabel
import scipy
from scipy.sparse.linalg import svds
import itertools 
import mvpa2
from mvpa2.suite import *
if __debug__:
    from mvpa2.base import debug
import os
os.chdir('//dartfs-hpc/rc/home/w/f003vpw/ObjectAnalysis/other')
from priorGPA_parallel import priorHyA
import pickle
import mvpa2.base.hdf5 as hd
from random import shuffle
#Functions


def distance_pairwise(ds_hyper):
    Xest = [ds.samples for ds in ds_hyper]
    diff = []
    for i in range(len(ds_hyper)):
       for j in range(len(ds_hyper)):
           if i != j & i<j:
               diff.append(Xest[i] - Xest[j])
    sum_distance = np.sum([np.linalg.norm(d, ord="fro")**2 for d in diff])
    
    return sum_distance
    
def traceProcrustes(ds_hyper):
    Xest = [ds.samples for ds in ds_hyper]
    M = np.mean(Xest, axis=0, dtype=np.float64)
    traceXM = [np.matrix.trace(x.T.dot(M)) for x in Xest]

    return traceXM


#Import data
ds_all = hd.h5load("//dartfs-hpc/rc/home/w/f003vpw/ObjectAnalysis/data/hyperalignment_tutorial_data_2.4.hdf5.gz")

__all__= ['priorHyA'] #explicitly exports the symbols priorHyA

# inject the subject ID into all datasets
for i, sd in enumerate(ds_all):
    sd.sa['subject'] = np.repeat(i, len(sd))
    
#be sure that we work in float
for ds in range(len(ds_all)):
    ds_all[ds].samples = ds_all[ds].samples.astype(np.float64)    

#type(ds_all[0].samples[0,0])

# number of subjects
nsubjs = len(ds_all)
# number of categories
ncats = len(ds_all[0].UT)
# number of run
nruns = len(ds_all[0].UC)

#We ll use a linear SVM classifier, and perform feature selection with a simple one-way ANOVA selecting the nf highest scoring features.

# use same classifier
clf = LinearCSVMC()

# feature selection helpers
nf = 100
fselector = FixedNElementTailSelector(nf, tail='upper',
                                      mode='select', sort=False)
sbfs = SensitivityBasedFeatureSelection(OneWayAnova(), fselector,
                                        enable_ca=['sensitivities'])
# create classifier with automatic feature selection
fsclf = FeatureSelectionClassifier(clf, sbfs)


"""
Anatomically aligned data without features selection
"""
mni_start_time = time.time()

ds_mni = vstack(ds_all)
cv = CrossValidation(fsclf, NFoldPartitioner(attr='subject'), errorfx=mean_match_accuracy)
bsc_mni_results = cv(ds_mni)
predictions = fsclf.predict(ds_mni.samples)
confusion = ConfusionMatrix()
confusion.add(ds_mni.targets, predictions)
cmA = confusion.matrix
mni_time = time.time() - mni_start_time

"""
Hyperaligment aligned data, to avoid circularity problem we perform leave-one-out in runs.
Subsequently, we will apply the derived transformation to the full datasets.
"""
dist = []
hyper_start_time = time.time()
cm_mean = []
bsc_hyper_results = []
cm = []
distH = []
traceH = []
i = 1 
out = []
tracePr = []
dist = []
idx = []
while i<101:
  shuffle(ds_all)
  for test_run in range(nruns):
      # split in training and testing set
      ds_train = [sd[sd.sa.chunks != test_run, :] for sd in ds_all]
      ds_test = [sd[sd.sa.chunks == test_run, :] for sd in ds_all]
  
      # manual feature selection for every individual dataset in the list
      anova = OneWayAnova()
      fscores = [anova(sd) for sd in ds_train]
      featsels = [StaticFeatureSelection(fselector(fscore)) for fscore in fscores]
      ds_train_fs = [fs.forward(sd) for fs, sd in zip(featsels, ds_train)]
      ds_test_fs = [fs.forward(sd) for fs, sd in zip(featsels, ds_test)]

      hyper = Hyperalignment()
      hypmaps = hyper(ds_train_fs)

      ds_hyper = [h.forward(sd) for h, sd in zip(hypmaps, ds_test_fs)]
      distH.append(distance_pairwise(ds_hyper))
      traceH.append(traceProcrustes(ds_hyper))
      ds_hyper = vstack(ds_hyper)
      
      zscore(ds_hyper, chunks_attr='subject')
      res_cv = cv(ds_hyper)
      
      predictions = clf.predict(ds_hyper.samples)
      confusion = ConfusionMatrix()
      confusion.add(ds_hyper.targets, predictions)
      cm.append(confusion.matrix)
      bsc_hyper_results.append(res_cv)
  
  dist.append(np.mean(hstack(distH)))
  tracePr.append(np.mean(traceH, axis = 0))
  out.append(np.mean(hstack(bsc_hyper_results)))
  cm_mean.append(np.mean(cm,axis=0)) 
  i = i + 1
  bsc_hyper_results = []
  idx.append([np.unique(sd.sa['subject']) for sd in ds_all])

#   
hyper_time = time.time() - hyper_start_time
#
#
np.savez('/dartfs-hpc/rc/home/w/f003vpw/ObjectAnalysis/Output/fs_cv_100_Hyper.npz', 
         out=out, dist = dist, tracePr = tracePr, idx = idx,
         cmA = cmA, cm_mean = cm_mean, 
         bsc_mni_results = bsc_mni_results,  
         distH = distH, hyper_time = hyper_time, mni_time = mni_time)


"""
We use Generalized Orthogonal Procrustes Analysis to align data, to avoid circularity problem we perform leave-one-out in runs.
Subsequently, we will apply the derived transformation to the full datasets.
"""

bsc_gpa0_results = []
dist0 = []
cm0 = []
dist0it = []
traceGPA = []
distGPA = []
#######################Generalized Procrustes Analysis without F###################

gpa_start_time = time.time()

for test_run in range(nruns):
    
    # split in training and testing set
    ds_train = [sd[sd.sa.chunks != test_run, :] for sd in ds_all]
    ds_test = [sd[sd.sa.chunks == test_run, :] for sd in ds_all]

    # manual feature selection for every individual dataset in the list
    anova = OneWayAnova()
    fscores = [anova(sd) for sd in ds_train]
    featsels = [StaticFeatureSelection(fselector(fscore)) for fscore in fscores]
    ds_train_fs = [fs.forward(sd) for fs, sd in zip(featsels, ds_train)]
    ds_test_fs = [fs.forward(sd) for fs, sd in zip(featsels, ds_test)]
                    
    #GPA normal
    
    hyper = priorHyA(maxIt = 20, t = 0.001, k = 0, Q = None, ref_ds = None,  scaling=True, reflection = True, subj=False)
    hypmaps = hyper.gpa(datasets=ds_train_fs)
    dist0it.append(hypmaps[3])
    ds_hyper = [h.forward(sd) for h, sd in zip(hypmaps[1], ds_test_fs)]
    distGPA.append(distance_pairwise(ds_hyper))
    traceGPA.append(traceProcrustes(ds_hyper))
    ds_hyper = mvpa2.base.dataset.vstack(ds_hyper)
    zscore(ds_hyper, chunks_attr='subject')
    res_cv = cv(ds_hyper)
    predictions = fsclf.predict(ds_hyper.samples)
    confusion = ConfusionMatrix()
    confusion.add(ds_hyper.targets, predictions)
    cm0.append(confusion.matrix)
    bsc_gpa0_results.append(res_cv)

gpa_time = time.time() - gpa_start_time
   
bsc_gpa0_results = hstack(bsc_gpa0_results)

mean_gpa0_results = np.mean(bsc_gpa0_results)

tracegpa = np.mean(traceGPA, axis = 0)

np.savez('/dartfs-hpc/rc/home/w/f003vpw/ObjectAnalysis/Output/fs_cv_100_GPA_S.npz', 
         distGPA=distGPA,  dist0it=dist0it, tracegpa = tracegpa,
         cm0 = cm0,  
         mean_gpa0_results = mean_gpa0_results, 
         gpa_time = gpa_time)

kval =  np.linspace(start=0.1, stop=10, num=5,dtype=float)

##############GPA with euclidean kernel F###############

bsc_gpaE_results = []
distE = []
cmE = []
bsc_mean_E = []
bsc_gpaE_results_test = []
distEit = []
#distEm = []
traceGPAprior = []
distGPAprior = []
gpaE_start_time = time.time()
distItprior = []
RGPAprior = []
Qsave = []
for test_run in range(nruns):
    
    # split in training and testing set
    ds_train = [sd[sd.sa.chunks != test_run, :] for sd in ds_all]
    ds_test = [sd[sd.sa.chunks == test_run, :] for sd in ds_all]

    # manual feature selection for every individual dataset in the list
    anova = OneWayAnova()
    fscores = [anova(sd) for sd in ds_train]
    featsels = [StaticFeatureSelection(fselector(fscore)) for fscore in fscores]
    ds_train_fs = [fs.forward(sd) for fs, sd in zip(featsels, ds_train)]
    ds_test_fs = [fs.forward(sd) for fs, sd in zip(featsels, ds_test)]
                        
    #GPA euclidean
    
    for k in kval:
        t_run = ds_train_fs[0].sa.chunks
        for t_cv in t_run:
              ds_val = [sd[sd.sa.chunks != t_cv, :] for sd in ds_train_fs]
              ds_t = [sd[sd.sa.chunks == t_cv, :] for sd in ds_train_fs]
              #GPA prior euclidean kernel
              coord = [d.fa.voxel_indices for d in ds_val]
              dist = [cdist(np.array(c), np.array(c), "euclidean") for c in coord]
              Q2 = [np.exp(-d/c.shape[0]) for d,c in zip(dist,coord)]
              hyper = priorHyA(maxIt = 20, t = 0.001, k = k, Q = Q2, ref_ds = None,  scaling=True, reflection = True, subj=True)
              hypmaps = hyper.gpa(datasets=ds_val)
              #distE.append(distance_pairwise(hypmaps= hypmaps, dss = ds_val, alignment='gpa'))
              ds_hyper = [h.forward(sd) for h, sd in zip(hypmaps[1], ds_t)]
              ds_hyper = mvpa2.base.dataset.vstack(ds_hyper)  
              zscore(ds_hyper, chunks_attr='subject')
              res_cv = cv(ds_hyper)
              bsc_gpaE_results.append(res_cv)
        
        #distEm.append(np.mean(distE))
        #distE = []
        bsc_gpaE_results = hstack(bsc_gpaE_results)
        bsc_mean_E.append(np.mean(bsc_gpaE_results))
        bsc_gpaE_results = []
    k_cvE = zip(kval,bsc_mean_E)
    khat = max(k_cvE, key = lambda t: t[1])[0]
    
    coord = [d.fa.voxel_indices for d in ds_train_fs]
    dist = [cdist(np.array(c), np.array(c), "euclidean") for c in coord]
    Q2 = [np.exp(-d/c.shape[0]) for d,c in zip(dist,coord)]
    Qsave.append(Q2)
    hyper = priorHyA(maxIt = 20, t = 0.001, k = khat, Q = Q2, ref_ds = None,  scaling=True, reflection = True, subj=True)
    hypmaps = hyper.gpa(datasets=ds_train_fs)
    distEit.append(hypmaps[3])          
    ds_hyper = [h.forward(sd) for h, sd in zip(hypmaps[1], ds_test_fs)]
    distGPAprior.append(distance_pairwise(ds_hyper))
    traceGPAprior.append(traceProcrustes(ds_hyper))
    RGPAprior.append(hypmaps[2])
    distItprior.append(hypmaps[3])
    ds_hyper = mvpa2.base.dataset.vstack(ds_hyper)
    zscore(ds_hyper, chunks_attr='subject')
    res_cv = cv(ds_hyper)
    predictions = fsclf.predict(ds_hyper.samples)
    confusion = ConfusionMatrix()
    confusion.add(ds_hyper.targets, predictions)
    cmE.append(confusion.matrix)
    bsc_gpaE_results_test.append(res_cv)

gpaE_time = time.time() - gpaE_start_time

cmE_mean = np.mean(cmE,axis=0)    
   
bsc_gpaE_results = hstack(bsc_gpaE_results_test)

mean_gpaE_results = np.mean(bsc_gpaE_results)

tracegpaprior = np.mean(traceGPAprior, axis = 0)

np.savez('/dartfs-hpc/rc/home/w/f003vpw/ObjectAnalysis/Output/fs_cv_100_GPAprior_S.npz',   
         cmE_mean = cmE_mean,  distEit = distEit, distGPAprior = distGPAprior, tracegpaprior = tracegpaprior,
         mean_gpaE_results = mean_gpaE_results,  distItprior = distItprior, Qsave = Qsave,
         gpaE_time = gpaE_time)

