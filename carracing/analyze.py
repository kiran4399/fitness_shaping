
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



data_a, data_b = load_muvar('generation/')


for i in range(len(data_a)):
    print(data_a[i,1])