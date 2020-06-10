#!/usr/bin/env python

"""
Final analysis LO using nested CV to choose k
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
os.chdir('//dartfs-hpc/rc/home/w/f003vpw/MovieAnalysis/Final_analysis')
#import geopy as gp
#from priorGPA_1 import priorHyA
from time_segment_cv import timesegments_classification
from function import distance_pairwise
import pickle
from random import shuffle
#import scipy

#ds_all =np.load('//dartfs-hpc/rc/home/w/f003vpw/MovieAnalysis/data/dataMovie.npz')["ds_all"]
#shape =np.load('//dartfs-hpc/rc/home/w/f003vpw/MovieAnalysis/data/dataMovie.npz')["shapeL"]
#surf_l =np.load('//dartfs-hpc/rc/home/w/f003vpw/MovieAnalysis/data/coord.npz')["surf_l"]
coord =np.load('//dartfs-hpc/rc/home/w/f003vpw/MovieAnalysis/data/coord_LO.npz')["coord"]
#Dk = np.load('//dartfs-hpc/rc/home/w/f003vpw/MovieAnalysis/report_data/dij.npz')['dij']

#load data
with open('//dartfs-hpc/rc/home/w/f003vpw/MovieAnalysis/data/movie_LO.data', 'rb') as movie:  
    # read the data as binary data stream
    dss = pickle.load(movie)
    
with open('//dartfs-hpc/rc/home/w/f003vpw/MovieAnalysis/data/movie32_LO.data', 'rb') as movie32:  
    # read the data as binary data stream
    dss1 = pickle.load(movie32)    
    
dss = dss + dss1

#Be sure that we work in float
for ds in range(len(dss)):
    dss[ds].samples = dss[ds].samples.astype(np.float64)  


#Anatomical
an = timesegments_classification(dss, hyper = None,
                            part1=HalfPartitioner(), 
                            part2=NFoldPartitioner(attr='subjects'), 
                            window_size=6, overlapping_windows=True, 
                            distance='correlation', do_zscore=True)


shuffle(dss)
idxs1 = [np.unique(sd.sa['subject']) for sd in dss]     
#Hyperaligment
hyps1, hypmapss1, distHs1, traceHs1 = timesegments_classification(dss, hyper = 'h',
                            part1=HalfPartitioner(), 
                            part2=NFoldPartitioner(attr='subjects'), 
                            window_size=6, overlapping_windows=True, 
                            distance='correlation', do_zscore=True)
shuffle(dss)
idxs2 = [np.unique(sd.sa['subject']) for sd in dss]     
#Hyperaligment
hyps2, hypmapss2, distHs2, traceHs2 = timesegments_classification(dss, hyper = 'h',
                            part1=HalfPartitioner(), 
                            part2=NFoldPartitioner(attr='subjects'), 
                            window_size=6, overlapping_windows=True, 
                            distance='correlation', do_zscore=True)
                            
shuffle(dss)
idxs3 = [np.unique(sd.sa['subject']) for sd in dss]     
#Hyperaligment
hyps3, hypmapss3, distHs3, traceHs3 = timesegments_classification(dss, hyper = 'h',
                            part1=HalfPartitioner(), 
                            part2=NFoldPartitioner(attr='subjects'), 
                            window_size=6, overlapping_windows=True, 
                            distance='correlation', do_zscore=True)                            
#Generalized Procrustes Analysis
gpa, hypmaps0, distH0, traceH0 = timesegments_classification(dss, hyper = 'gpa',
                            part1=HalfPartitioner(), 
                            part2=NFoldPartitioner(attr='subjects'), 
                            window_size=6, overlapping_windows=True, 
                            distance='correlation', do_zscore=False, maxIt=10, kval=0, Q=None,ref_ds=None,t=0.001, scaling=True,
                            reflection = True, subj=False)

#Generalized Procrustes Analysis with prior

#We selected 5 type of Q: mds, Dijkstra, euclidean, cosine, cityblock, mean

#kval = np.hstack([np.linspace(start=0, stop=100, num=20,dtype=int)] + [np.linspace(start=100, stop=10000, num=10,dtype=int)])
kval =  np.linspace(start=0.1, stop=20, num=10,dtype=float)


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
                            maxIt = 10, t = 0.001, Q = Q2, ref_ds = None,  scaling=False, reflection = True, subj=False)

Q2 = np.diag(np.ones(dist.shape[0]))

errI, hypmapsI, kCVI, distHI, traceHI, RGPApriorI, distItpriorI = timesegments_classification(dss, hyper = 'gpa',
                            part1=HalfPartitioner(), 
                            part2=NFoldPartitioner(attr='subjects'), 
                            window_size=6, overlapping_windows=True, 
                            distance='correlation', do_zscore=False,
                            maxIt = 10, t = 0.001, kval = kval, Q = Q2, ref_ds = None,  scaling=False, reflection = True, subj=False)

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



np.savez('//dartfs-hpc/rc/home/w/f003vpw/MovieAnalysis/Output/Final_analysis_LO_cv.npz', 
         errE=errE, errI = errI, an = an, hyp = hyp, gpa = gpa, kval = kval, 
#         errD = errD,
         kCVE = kCVE, 
         kCVI = kCVI, idx1 = idx1, idx2 = idx2, idx3 = idx3,
#         kCVD = kCVD,
         hyp_pair = hyp_pair, gpa_pair = gpa_pair, gpaE_pair = gpaE_pair, gpaI_pair = gpaI_pair, 
#         gpaD_pair = gpaD_pair,
         distHE = distHE, traceHE = traceHE, RGPApriorE = RGPApriorE, distItpriorE = distItpriorE,
#         distHD, traceHD, RGPApriorD, distItpriorD,
         distHI = distHI, traceHI = traceHI, RGPApriorI = RGPApriorI, distItpriorI = distItpriorI,
         distH0 = distH0, traceH0 = traceH0,
         distHs1 = distHs1, traceHs1=traceHs1, distHs2=distHs2, traceHs2=traceHs2, distHs3=distHs3, traceHs3=traceHs3
         )


    
    
