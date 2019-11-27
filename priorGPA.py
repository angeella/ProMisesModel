
#!/usr/bin/env python

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

__all__= ['priorGPA'] #explicitly exports the symbols priorHyA

#if you need explit here some function as 

class priorGPA(ClassWithCollections):
    
    """Align multiple matrices, e.g., fMRI images, into a common feature space using Procrustes orthogonal transformation. 
    The aligorithm is a slight modification of the Generalized Procrustes Analysis :ref:`Gower, J. C., Psychometrika, (1975).` 
    *Generalized procrustes analysis.* We decompose by Singular Value Decomposition X_i^\top M + k Q instead of X_i^\top M.
    Examples
    --------
    
    >>> import numpy as np
    >>> import mvpa2
    >>> from mvpa2.suite import *
    >>> import os
    >>> os.chdir('yourpathwhere_priorGPA.py_is')
    >>> from priorGPA import priorGPA
    >>> #Load dataset
    >>> ds_all = h5load('//dartfs-hpc/rc/home/w/f003vpw/ObjectAnalysis/data/hyperalignment_tutorial_data_2.4.hdf5.gz')
    >>> #Number of voxels to select
    >>> nf = 80
    >>> # Automatic feature selection
    >>> fselector = FixedNElementTailSelector(nf, tail='upper',mode='select', sort=False)
    >>> anova = OneWayAnova()
    >>> fscores = [anova(sd) for sd in ds_all]
    >>> featsels = [StaticFeatureSelection(fselector(fscore)) for fscore in fscores]
    >>> ds_all_fs = [fs.forward(sd) for fs, sd in zip(featsels, ds_all)]
    >>> #Compute Location Parameter as Similarity Matrix using the euclidean distance of the three dimensional coordinates of the voxels
    >>> coord = [ds.fa.voxel_indices for ds in ds_all_fs]
    >>> dist = [cdist(c, c, "euclidean") for c in coord]
    >>> Q = [np.exp(-d/c.shape[0]) for d,c in zip(dist,coord)]
    >>> #Alignment step
    >>> gp = priorGPA(maxIt = 10, t = 0.001, k = 1, Q = Q, ref_ds = None,  scaling=True, reflection = True, subj=True)
    >>> mappers = gp.gpa(ds_all_fs)[1]
    >>> len(mappers)
    """
    
    #context = getcontext()
    #context.prec = 55
    #chosen_ref_ds = mvpa2.ConditionalAttribute(enabled=True,
            #doc="""Index of the input dataset used as 1st-level reference dataset.""")
    
    maxIt = Parameter(5, constraints=EnsureInt() & EnsureRange(min=0)),
            doc=""" maximum number of iteration used in the Generalised Procrustes Analysis """)
    
    t = Parameter(1, constraints= EnsureRange(min=0), doc=""" the threshold value to be reached as the minimum relative reduction between the matrices """)
    
    k = Parameter(1, constraints= EnsureRange(min=0), doc=""" value of the concentration parameter of the prior distribution """)
    
    Q = Parameter(0, doc=""" value of the location parameter of the prior distribution. It has dimension voxels x voxels, it could be not symmetric. """)
    
    ref_ds = mvpa2.Parameter(None, doc=""" index starting matrix to align """)
    
    scaling = Parameter(True, constraints='bool',
              doc="""Flag to apply scaling transformation""")
    
    reflection = Parameter(True, constraints='bool',
                 doc="""Flag to apply reflection transformation""")
    
    subj = Parameter(True, constraints='bool',
                 doc="""Flag if each subject has his/her own set of voxel after voxel selection step""")
    
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
        if not isinstance(datasets, (list, tuple, np.ndarray)):
            raise TypeError("Input datasets should be a sequence "
                           "(of type list, tuple, or ndarray) of datasets.")
        
        ndatasets = len(datasets)
        #quick access parameters
        k = self.k 
        Q = self.Q
        t = self.t
        maxIt = self.maxIt
        ref_ds = self.ref_ds
        scaling = self.scaling
        reflection = self.reflection
        subj = self.subj

        #Implement having list without datasets structure
        shape_datasets = [ds.samples.shape for ds in datasets]
        
        if not all(x==shape_datasets[0] for x in shape_datasets):
            raise ValueError('the shapes of datasets are different')
        
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
                    U, s, Vt =  np.linalg.svd((ref_ds.T.dot(X[i]) + k * Q.T).T, full_matrices = False)
                    #U, s, Vt =  np.linalg.svd(X[i].T.dot(ref_ds) + k * Q, full_matrices = False)

                else:
                    if Q is None:
                        Q = np.zeros(col) 
                    U, s, Vt =  np.linalg.svd((ref_ds.T.dot(X[i]) + k * Q[i].T).T, full_matrices = False)      
                    #U, s, Vt =  np.linalg.svd(X[i].T.dot(ref_ds) + k * Q[i], full_matrices = False)                

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
    
    
