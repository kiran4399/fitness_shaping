
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import json
import os
from mpi4py import MPI
import sys
import subprocess

np.set_printoptions(threshold=np.nan)

def load_muvar(folder):
    allfolders = os.listdir(folder)
    for each in allfolders:

        if each == allfolders[0]:
            data_a = np.load(folder+each)['mu']
            data_b = np.load(folder+each)['logvar']
        else:
            data_a = np.concatenate((data_a, np.load(folder+each)['mu']))
            data_b = np.concatenate((data_b, np.load(folder+each)['logvar']))

    return data_a, data_b



hdata_a, hdata_b = load_muvar('data/human/')
adata_a, adata_b = load_muvar('data/agent/')
print("num", "a", "b", "c", "d")
for i in range(200, len(hdata_b)-200):
    print(i, adata_b[i,0], hdata_b[i,0], adata_b[i,1], hdata_b[i,1])