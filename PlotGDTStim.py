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
from scipy.io.wavfile import write

fig_loc = 'C:/Users/vmysorea/Desktop/PhD/Conferences/ARO 2024/Chin_miniEEG/'

###Plotting GDT stim
# dat = io.loadmat('D:/PhD/Heinz Lab/VMA_GDT_Chins/VMA_Chin_GDT_EEG/ChinGDTStims_64ms.mat', 
#                  squeeze_me=True)
# dat1 = io.loadmat('C:/Users/vmysorea/Desktop/PhD/Stim_Analysis/GapDetection_EEG/Stimuli_Human/GDT_HumanStim32.mat', 
#                  squeeze_me=True)
dat2 = io.loadmat('C:/Users/vmysorea/Desktop/PhD/Stim_Analysis/GapDetection_EEG/Stimuli_Human/GDT_HumanStim_250.mat', 
                  squeeze_me=True)
# dat.keys()
# dat1.keys()
dat2.keys()
fs = np.int64(dat2['Fs'])
 
x2= dat2['x'] 
x2= np.int64(x2) 
# x=dat['stims']
# x= np.pad(dat['x'], (49609,51953))
# x1= dat1['x'] 

# t=np.arange(x[0].shape[0])/fs
# t1=np.arange(x1.shape[0])/fs
t2=np.arange(x2.shape[0])/fs
t2 = np.int64(t2)

# s = x+x1+x2

plt.plot(t2,x2, color='dimgrey')
plt.axvline(1.064, linestyle='dashed',color='blue', linewidth=3)
# plt.axvline(1.128, linestyle='dashed',color='blue', linewidth=2)
# plt.axvline(1.692, linestyle='dashed',color='blue', linewidth=2)
plt.ylabel('Stim Amplitude', fontsize=26)
plt.xlabel('Time (in s)', fontsize = 26)
# plt.title('Example of 64 ms Gap Trial', fontsize=22)
plt.text(0.8,0.29, '64 ms', fontsize=25, color='blue', weight='bold')
plt.xticks(fontsize=20)
plt.yticks([],[])
plt.tight_layout()
plt.show()

write('C:/Users/vmysorea/Desktop/PhD/Stim_Analysis/GapDetection_EEG/Stimuli_Human/GDTHuman_64ms.wav', fs, x2)

plt.savefig(fig_loc + 'ChinExample64ms.png',dpi=500, width=6, height=5, transparent=True)
