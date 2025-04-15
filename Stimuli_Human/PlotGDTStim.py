# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 21:53:01 2023

@author: vmysorea
"""

from matplotlib import pyplot as plt
from scipy.io import savemat
from scipy import io
import numpy as np 
from scipy import stats 
plt.rcParams['agg.path.chunksize'] = 10000


###Plotting GDT stim
dat = io.loadmat('C:/Users/vmysorea/Desktop/PhD/Stim_Analysis/GapDetection_EEG/Stimuli_Human/GDT_HumanStim16.mat', 
                 squeeze_me=True)
dat1 = io.loadmat('C:/Users/vmysorea/Desktop/PhD/Stim_Analysis/GapDetection_EEG/Stimuli/Human_GDT_EEG/GDT_HumanStim32.mat', 
                 squeeze_me=True)
dat2 = io.loadmat('C:/Users/vmysorea/Desktop/PhD/Stim_Analysis/GapDetection_EEG/Stimuli/Human_GDT_EEG/GDT_HumanStim64.mat', 
                 squeeze_me=True)
dat.keys()
dat1.keys()
dat2.keys()
fs = dat['Fs']
 
x2= dat2['x']  
x= np.pad(dat['x'], (49609,51953))
x1= dat1['x'] 

t=np.arange(x.shape[0])/fs
t1=np.arange(x1.shape[0])/fs
t2=np.arange(x2.shape[0])/fs

s = x+x1+x2



plt.plot(t,x)

