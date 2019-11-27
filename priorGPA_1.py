
#!/usr/bin/env python


# -*- coding: utf-8 -*-
"""
Perform hyperaligment with prior. The modification is in the SVD in the procustean problem.
We decompose X_i M + k Q instead of X_i M.
"""

#Import mvpa2 package
import numpy as np
import mvpa2
#import scipy as syp
from mvpa2.suite import * #makes everything directly accessible through the mvpa namespace.
from mvpa2.base import *
import scipy.linalg
from mvpa2.mappers.staticprojection import StaticProjectionMapper
#import pdb
import decimal as dc
#context = getcontext()
#context.prec = 55
import math

__all__= ['priorHyA'] #explicitly exports the symbols priorHyA

#if you need explit here some function as 

class priorHyA:
    
    #context = getcontext()
    #context.prec = 55
    #chosen_ref_ds = mvpa2.ConditionalAttribute(enabled=True,
            #doc="""Index of the input dataset used as 1st-level reference dataset.""")
    
    #maxIt = mvpa2.Parameter(5, constraints=(mvpa2.EnsureInt() & mvpa2.EnsureRange(min=0)),
            #doc=""" maximum number of iteration used in the Generalised Procrustes Analysis """)
    
    #T = mvpa2.Parameter(1, constraints= mvpa2.EnsureRange(min=0), doc=""" the threshold value to be reached as the minimum relative reduction between the matrices """)
    
    #k = mvpa2.Parameter(1, constraints= mvpa2.EnsureRange(min=0), doc=""" value of the concentration parameter of the prior distribution """)
    
    #Q = mvpa2.Parameter(0, doc=""" value of the location parameter of the prior distribution. It has dimension voxels x voxels, it could be not symmetric. """)
    
    #ref_ds = mvpa2.Parameter(None, doc=""" index starting matrix to align """)
    
    def __init__(self, maxIt, t, k, Q, ref_ds, scaling, reflection, subj):
        #mvpa2.base.state.ClassWithCollections.__init__(self)
        self.maxIt = maxIt
        self.t = t
        self.k = k
        self.Q = Q
        self.ref_ds = ref_ds
        self.scaling = scaling
        self.reflection = reflection
        self.subj = subj
    def gpa(self, datasets):
         
        #params = self.params            
        # Check to make sure we get a list of datasets as input.
        #if not isinstance(datasets, (list, tuple, np.ndarray)):
        #    raise TypeError("Input datasets should be a sequence "
        #                   "(of type list, tuple, or ndarray) of datasets.")
        
        ndatasets = len(datasets)
        #v = np.int64(datasets[0].samples.shape[1])
        #ssqs = ssqs = [np.sum(dataset.samples**2, axis=0) for d in datasets]
        #norms = [ np.sqrt(np.sum(ssq)) for ssq in ssqs ]
        #normed = [ data.samples/norm for (data, norm) in zip(datasets, norms) ]
        #nfeatures = [ds.nfeatures for ds in datasets]
        k = self.k #quick access
        Q = self.Q
        t = self.t
        maxIt = self.maxIt
        ref_ds = self.ref_ds
        scaling = self.scaling
        reflection = self.reflection
        subj = self.subj
        #if A.ndim != 2:
        #    raise ValueError('expected ndim to be 2, but observed %s' % A.ndim)
        
        #if A.shape != B.shape:
        #raise ValueError('the shapes of A and B differ (%s vs %s)' % (
        #    A.shape, B.shape))
        #A = (datasets[ds].samples for ds in range(ndatasets))
        
        row, col = datasets[0].samples.shape
        
        count = 0
        dist = []
        dist.append(np.inf)
        datas = [dt.samples for dt in datasets]
        datas_centered = [d - np.mean(d, 0) for d in datas]
        
        norms = [np.linalg.norm(dce) for dce in datas_centered]
        
        if np.any(norms == 0):
            raise ValueError("Input matrices must contain >1 unique points")
        
        X = [dce/n for dce,n in zip(datas_centered,norms)]
        
        #X = [dt.samples for dt in datasets]
        
        if ref_ds is None:
            #ref_ds = np.mean([datasets[ds].samples for ds in range(ndatasets)], axis = 0)
            ref_ds = np.mean(X, axis=0, dtype=np.float64)
        
        while dist[count] > t and count < maxIt:
            Xest = []
            R = []
            #ref_start = ref_ds
            #del ref_ds
            for i in range(ndatasets):
                #M = ref_ds
                #Xs = M.T + k * Q 
                if not subj:
                    if Q is None:
                        Q = np.zeros((col,col)) 
                    #Put transposes to save memory.
                    #U, s, Vt =  np.linalg.svd((ref_ds.T.dot(X[i]) + k * Q.T).T, full_matrices = False)
                    U, s, Vt =  np.linalg.svd(X[i].T.dot(ref_ds) + k * Q, full_matrices = False)

                else:
                    if Q is None:
                        Q = np.zeros(col) 
                    #U, s, Vt =  np.linalg.svd((ref_ds.T.dot(X[i]) + k * Q[i].T).T, full_matrices = False)      
                    U, s, Vt =  np.linalg.svd(X[i].T.dot(ref_ds) + k * Q[i], full_matrices = False)                

                if not reflection: 
                    s_new = np.diag(np.ones(len(s)))
                    s_new[-1,-1] = np.sign(np.linalg.det(U.dot(Vt)))
                    Tr = U.dot(s_new).dot(Vt)
                    scale = np.sum(s_new * s)
                else:
                    Tr = U.dot(Vt)
                    scale = np.sum(s)
                
                R.append(Tr)
                #R[i] = U.dot(Vh)
                if not scaling:
                    Xest.append(X[i].dot(R[i]))
                else:
                    Xest.append((X[i].dot(R[i]))* scale) 
                #Xest[i] = Xest[i].dot(R[i])
            count +=1
            ref_ds_old = np.copy(ref_ds)
            #print(ref_ds_old)
            ref_ds = np.mean(Xest, axis=0)
            #print(ref_ds)
            #ref_ds = sum(Xest[ds] for ds in range(ndatasets))/ndatasets
            diff = np.subtract(ref_ds,ref_ds_old, dtype=np.float64)
            dist.append(np.linalg.norm(diff, ord='fro'))
        
        
        
        rot = [mvpa2.mappers.staticprojection.StaticProjectionMapper(np.matrix(R[p]),auto_train=False) for p in range(ndatasets)]
        return Xest, rot, R, dist, ref_ds, ref_ds_old, datasets, count
    
__version__ = '0.1'    
    
    
if __name__ == '__main__':
    # test1.py executed as script
    # do something
    priorHyA()    
    
    