#!/usr/bin/env python

"""
VT Raiders Analysis using nested cross-validation 
"""

import numpy as np
#import nibabel
import mvpa2
from mvpa2.suite import *
#from scipy.sparse.linalg import svds
#import matplotlib.pyplot as plt
#import itertools
import os
import numpy as np
from mvpa2.base import warning
from mvpa2.support import copy
from mvpa2.mappers.base import IdentityMapper
from mvpa2.mappers.zscore import zscore
from mvpa2.generators.partition import NFoldPartitioner, HalfPartitioner
from mvpa2.generators.splitters import Splitter
from mvpa2.mappers.boxcar import BoxcarMapper
from mvpa2.datasets.base import FlattenMapper
from mvpa2.measures.anova import vstack
from mvpa2.mappers.fx import mean_group_sample
os.chdir('C:/Users/Angela Andreella/Documents/GitHub/vMFPmodel') #your path
from vMFPmodel import vMFPmodel
from time_segment_cv import timesegments_classification
from function import distance_pairwise
import pickle
from random import shuffle
#import scipy

#Load coordinates
coord =np.load('/Data/Raiders/Raiders_data/coord.npz')["coord"]
#Dk = np.load('//dartfs-hpc/rc/home/w/f003vpw/MovieAnalysis/report_data/dij.npz')['dij']

#Load data
with open('/Data/Raiders/Raiders_data/movie.data', 'rb') as movie:  
    # read the data as binary data stream
    dss = pickle.load(movie)
    
with open('/Data/Raiders/Raiders_data/movie32.data', 'rb') as movie32:  
    # read the data as binary data stream
    dss1 = pickle.load(movie32)    
    
dss = dss + dss1

#Be sure that we work in float
for ds in range(len(dss)):
    dss[ds].samples = dss[ds].samples.astype(np.float64)  
    
#Anatomical alignment
an = timesegments_classification(dss, hyper = None,
                            part1=HalfPartitioner(), 
                            part2=NFoldPartitioner(attr='subjects'), 
                            window_size=6, overlapping_windows=True, 
                            distance='correlation', do_zscore=True)


i = 0
Hyps = []
Hyps= []
Hypmaps= []
DistH= []
TraceH= []

#Hyperaligment
while i<101:
  shuffle(dss)
    

  hyps, hypmaps, distH, traceH = timesegments_classification(dss, hyper = 'h',
                                 part1=HalfPartitioner(), 
                                 part2=NFoldPartitioner(attr='subjects'), 
                                 window_size=6, overlapping_windows=True, 
                                 distance='correlation', do_zscore=True)
                                 
  Hyps.append(hyps)
  Hypmaps.append(hypmaps)
  DistH.append(distH)
  TraceH.append(traceH)
  i = i + 1
 
mean_perm = hstack(Hyps)

#Generalized Procrustes Analysis
gpa, hypmaps0, distH0, traceH0 = timesegments_classification(dss, hyper = 'gpa',
                            part1=HalfPartitioner(), 
                            part2=NFoldPartitioner(attr='subjects'), 
                            window_size=6, overlapping_windows=True, 
                            distance='correlation', do_zscore=False, maxIt=10, kval=0, Q=None,ref_ds=None,t=0.001, scaling=True,
                            reflection = True, subj=False)

#von Mises Fisher Procrustes model

kval =  np.linspace(start=0.1, stop=100, num=100,dtype=float)

n = float(coord.shape[0])

nsubjs = len(dss)

#Euclidean kernel
dist = cdist(np.array(coord), np.array(coord), "euclidean") 
Q2 = np.exp(-dist/n)

errE, hypmapsE, kCVE, distHE, traceHE, RGPApriorE, distItpriorE = timesegments_classification(dss, hyper = 'gpa',
                            part1=HalfPartitioner(), 
                            part2=NFoldPartitioner(attr='subjects'), 
                            window_size=6, overlapping_windows=True, 
                            distance='correlation', do_zscore=False, kval = kval,
                            maxIt = 10, t = 0.001, Q = Q2, ref_ds = None,  scaling=True, reflection = True, subj=False)

#Identity kernel
Q2 = np.diag(np.ones(dist.shape[0]))

errI, hypmapsI, kCVI, distHI, traceHI, RGPApriorI, distItpriorI = timesegments_classification(dss, hyper = 'gpa',
                            part1=HalfPartitioner(), 
                            part2=NFoldPartitioner(attr='subjects'), 
                            window_size=6, overlapping_windows=True, 
                            distance='correlation', do_zscore=False,
                            maxIt = 10, t = 0.001, kval = kval, Q = Q2, ref_ds = None,  scaling=True, reflection = True, subj=False)

#Q2 = Dk

#errD, hypmapsD, kCVD, distHD, traceHD, RGPApriorD, distItpriorD = timesegments_classification(dss, hyper = 'gpa',
#                            part1=HalfPartitioner(), 
#                            part2=NFoldPartitioner(attr='subjects'), 
#                            window_size=6, overlapping_windows=True, 
#                            distance='correlation', do_zscore=False,
#                            maxIt = 10, t = 0.001, kval = kval, Q = Q2, ref_ds = None,  scaling=True, reflection = True, subj=False)

    
#Compute pairwise distance

hyp_pair = distance_pairwise(hypmaps = hypmaps, dss = dss)
gpa_pair = distance_pairwise(hypmaps = hypmaps0, dss = dss)
gpaE_pair = distance_pairwise(hypmaps = hypmapsE, dss = dss)
gpaI_pair = distance_pairwise(hypmaps = hypmapsI, dss = dss)
#gpaD_pair = distance_pairwise(hypmaps = hypmapsD, dss = dss)



np.savez('Analysis_VT.npz', 
         errE=errE, errI = errI, an = an, mean_perm = mean_perm, gpa = gpa, 
         hyp_pair = hyp_pair, gpa_pair = gpa_pair, gpaE_pair = gpaE_pair, gpaI_pair = gpaI_pair, 
         distHE = distHE, traceHE = traceHE, RGPApriorE = RGPApriorE, distItpriorE = distItpriorE,
         distHI = distHI, traceHI = traceHI, RGPApriorI = RGPApriorI, distItpriorI = distItpriorI,
         distH0 = distH0, traceH0 = traceH0, distH = DistH, traceH=TraceH)



    
