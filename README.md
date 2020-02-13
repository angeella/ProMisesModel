# Generalized Procrustes Analysis with Prior Information
DOI: 10.5281/zenodo.3629269

The method is implemented in Python according to the [PyMVPA](http://www.pymvpa.org/index.html) (MultiVariate Pattern Analysis in Python) package. 

First of all, the Generalized Procrustes Analysis with Prior information, i.e. spatial anatomical brain information, allows to improve the between-subject analysis in fMRI data. It is a useful functional alignment to perfom if someone want to analyze a set of subject. It must be used after data preprocessing, as motion correction, anatomical alignment and so on. The method is computationally heavy, at the moment, for that we recommend to use a mask from FSL, Matlab or other software. The mask must be used without falling into the double dipping problem. So, you must choose the mask after seeing the data, having some prior knowledge about the correlation between the region of interest and the type of task-design activation.

We are working also on a new method that permits to align the whole brain, so.. stay tuned! :)

Let's start with this short tutorial to understand easily how apply this functional alignment.

First of all, you need to install Python, and the PyMVPA, decimal, math, multiprocessing packages. If it is your first time with Python, you can see my [repository](https://github.com/angeella/Python_Tutorial) where I explain how install packages, how manage Anaconda and so on.

So, you need to download the script [priorGPA.py](https://github.com/angeella/priorGPA/blob/master/priorGPA.py) and put on the folder where you are working. 

After the installation, you need to import the packages:
```python
import numpy as np
import mvpa2
from mvpa2.suite import *
import os
```
Setup your path:

```python
os.chdir('your_path_where_priorGPA.py_is')
```
Then, load the datasets that must be [datasets from PyMVPA](http://www.pymvpa.org/tutorial_datasets.html?highlight=datasets) or a np.ndarray.

```python
os.chdir('your_path_where_priorGPA.py_is')
```
