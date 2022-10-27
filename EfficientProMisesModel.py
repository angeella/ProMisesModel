# -*- coding: utf-8 -*-
"""
Efficient ProMises in Python 3 (only identity)
"""

#Import mvpa2 package
import numpy as np
import multiprocessing as mp
import scipy.spatial.distance

__all__= ['EfficientProMisesModel'] #explicitly exports the symbols EfficientProMisesModel

#Internal function
def gpa(X, Q, k, ref_ds, col, scaling, reflection):

    if Q is None:
        Q = np.zeros((col,col)) 
    #Put transposes to save memory.
    U, s, Vt =  np.linalg.svd((ref_ds.T.dot(X) + k * Q.T).T, full_matrices = False)    
          
    if not reflection: 
        s_new = np.diag(np.ones(len(s)))
        s_new[-1,-1] = np.sign(np.linalg.det(U.dot(Vt)))
        Tr = U.dot(s_new).dot(Vt)
        scale = np.sum(s_new * s)
    else:
        Tr = U.dot(Vt)
        scale = np.sum(s)
                
    R = Tr

    if not scaling:
        Xest = X.dot(R)
    else:
        Xest = X.dot(R)* scale
    return Xest, R, scale



class EfficientProMisesModel:
    
    def __init__(self, maxIt, t, k, Q, ref_ds, scaling, reflection, subj, centered, all_info):
        #mvpa2.base.state.ClassWithCollections.__init__(self)
        self.maxIt = maxIt
        self.t = t
        self.k = k
        self.Q = Q
        self.ref_ds = ref_ds
        self.scaling = scaling
        self.reflection = reflection
        self.subj = subj
        self.centered = centered
        self.all_info = all_info
    def EfficientProMisesModel(self, datasets):
         
        k = self.k #quick access
        Q = self.Q
        t = self.t
        maxIt = self.maxIt
        ref_ds = self.ref_ds
        scaling = self.scaling
        reflection = self.reflection
        subj = self.subj
        centered = self.centered
        all_info = self.all_info
        
        count = 0
        dist = []
        dist.append(np.inf)
        
        #Semi-orthogonal transformation

        XQ = []
        Qlist = []
        for ds in range(len(datasets)):
          U, S, Q = np.linalg.svd(datasets[ds], full_matrices = False) 
          Qlist.append(Q)
          XQ.append(datasets[ds].dot(Q.T))
        
        datasets = XQ
        del XQ
        
        if centered:
            datasets = [d - np.mean(datasets, 0) for d in datasets]
        
        norms = [np.linalg.norm(dce) for dce in datasets]
        
        if np.any(norms == 0):
            raise ValueError("Input matrices must contain >1 unique points")
        
        X = [dce/n for dce,n in zip(datasets,norms)]
        
        del datasets
        del norms 
        #X = [dt.samples for dt in datasets]
        
        if ref_ds is None:
            #ref_ds = np.mean([datasets[ds].samples for ds in range(ndatasets)], axis = 0)
            ref_ds = np.mean(X, axis=0, dtype=np.float64)
        Xest = X
        del X
        while dist[count] > t and count < maxIt:
            #Xest = []
            #R = []
            #ref_start = ref_ds
            #del ref_ds
            #U, S, F = np.linalg.svd(ref_ds, full_matrices = False) 
            #F = F.T
            #del U
            #del S
            #Xstar = [x.dot(F) for x in X]
            #ref_ds_star = ref_ds.dot(F)
            row, col = Xstar[0].shape
            
            Q = np.matrix(np.identity(col)) * (col - row)/row
            
            pool = mp.Pool(mp.cpu_count())
            if subj:
                out = [pool.apply(gpa, args=(x, q, k, ref_ds, col, scaling, reflection)) for x,q in zip(Xest,Q)]
            else:
                out = [pool.apply(gpa, args=(x, Q, k, ref_ds, col, scaling, reflection)) for x in Xest]
            pool.close()  
            count +=1
            Xest = [x[0] for x in out]
          #  Rstar = [x[1] for x in out]
          #  scale = [x[2] for x in out]
          #  R = [F.dot(r).dot(F.T) for r in Rstar]
          #  Xest = [x.dot(F.T) for x in Xeststar]
            ref_ds_old = np.copy(ref_ds)
            #print(ref_ds_old)
            ref_ds = np.mean(Xest, axis=0)
            #print(ref_ds)
            #ref_ds = sum(Xest[ds] for ds in range(ndatasets))/ndatasets
            diff = np.subtract(ref_ds,ref_ds_old, dtype=np.float64)
            dist.append(np.linalg.norm(diff, ord='fro'))               
        
       # R1 = [r*s for r,s in zip(R,scale)]
       # rot = [mvpa2.mappers.staticprojection.StaticProjectionMapper(np.matrix(R1[p]),auto_train=False) for p in range(ndatasets)]
       # XestLight = Xeststar
        R = [x[1] for x in out]
        XestQ = []
        for d in range(len(Xest)):
            XestQ[d] = Xest[d].dot(Qlist[d])
        Xest = XestQ
        del XestQ
        if all_info is True:
            return Xest, R, dist, ref_ds, ref_ds_old, count, F
        else:
            return Xest
        
    
__version__ = '0.1'    
    
    
if __name__ == '__main__':
    # test1.py executed as script
    # do something
    EfficientProMisesModel() 
