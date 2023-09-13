# -*- coding: utf-8 -*-
"""
Created on Fri Jul 14 21:16:01 2023

@author: vmysorea
"""
##Plotting across gaps, across chinchillas

import sys
sys.path.append('C:/Users/vmysorea/Documents/mne-python/')
import warnings
from matplotlib import pyplot as plt
from scipy import io
import numpy as np
from scipy.stats import sem

plt.switch_backend('QT5Agg')  # Making the plots interactive (Scrollable)
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')
# Defining the dimensions and quality of figures
plt.rcParams["figure.figsize"] = (5.5,5)
plt.rcParams['figure.dpi'] = 120
#%%Setting up stuff
save_loc = ('C:/Users/vmysorea/Desktop/PhD/Stim_Analysis/GapDetection_EEG/Analysis_Chinchilla/')

subjlist = ['Q428', 'Q410']

#%%Loading mat files -- Need to work on this later

for subj in range(len(subjlist)):
    sub = subjlist[subj]
    dat1 = io.loadmat(save_loc + sub + '_16ms.mat', squeeze_me=True)
    dat2 = io.loadmat(save_loc + sub + '_32ms.mat', squeeze_me=True)
    dat3 = io.loadmat(save_loc + sub + '_64ms.mat', squeeze_me=True)

    dat1.keys()
    dat2.keys()
    dat3.keys()
    
    ep_mastoid16 = dat1['ep_mastoid']
    ep_vertex16 = dat1['ep_vertex']
    ep_ground16= dat1['ep_ground']
    ep_all16 = dat1['ep_all']
    ep_mean16 = dat1['ep_mean']
    ep_sem16 = dat1['ep_sem']
    ep_subderm16 = dat1['ep_subderm']
    ep_mean_subderm16 = dat1['ep_mean_subderm']
    ep_sem_subderm16 = dat1['ep_sem_subderm']
    picks= dat1['picks']
    t=dat1['t']
    
    ep_mastoid32 = dat2['ep_mastoid']
    ep_vertex32 = dat2['ep_vertex']
    ep_ground32= dat2['ep_ground']
    ep_all32 = dat2['ep_all']
    ep_mean32 = dat2['ep_mean']
    ep_sem32 = dat2['ep_sem']
    ep_subderm32 = dat2['ep_subderm']
    ep_mean_subderm32 = dat2['ep_mean_subderm']
    ep_sem_subderm32 = dat2['ep_sem_subderm']
    
    ep_mastoid64 = dat3['ep_mastoid']
    ep_vertex64 = dat3['ep_vertex']
    ep_ground64= dat3['ep_ground']
    ep_all64 = dat3['ep_all']
    ep_mean64 = dat3['ep_mean']
    ep_sem64 = dat3['ep_sem']
    ep_subderm64 = dat3['ep_subderm']
    ep_mean_subderm64 = dat3['ep_mean_subderm']
    ep_sem_subderm64 = dat3['ep_sem_subderm']
    
vertex16 = ep_vertex16.mean(axis=0)
vertex_sem16 = ep_vertex16.std(axis=0) / np.sqrt(ep_vertex16.shape[0])
mastoid16 = ep_mastoid16.mean(axis=0)
mastoid_sem16 = ep_mastoid16.std(axis=0) / np.sqrt(ep_mastoid16.shape[0])

vertex32 = ep_vertex32.mean(axis=0)
vertex_sem32 = ep_vertex32.std(axis=0) / np.sqrt(ep_vertex32.shape[0])
mastoid32 = ep_mastoid32.mean(axis=0)
mastoid_sem32 = ep_mastoid32.std(axis=0) / np.sqrt(ep_mastoid32.shape[0])

vertex64 = ep_vertex64.mean(axis=0)
vertex_sem64 = ep_vertex64.std(axis=0) / np.sqrt(ep_vertex64.shape[0])
mastoid64 = ep_mastoid64.mean(axis=0)
mastoid_sem64 = ep_mastoid64.std(axis=0) / np.sqrt(ep_mastoid64.shape[0])

###Plotting 

fig, ax = plt.subplots(2, 1, sharex=True, sharey=True)
ax[0].errorbar(t, ep_mean16, yerr=ep_sem16,color='green', linewidth=1, ecolor='lightgreen',label='16 ms' +str(picks))
ax[0].errorbar(t, ep_mean32, yerr=ep_sem32,color='purple', linewidth=1, ecolor='thistle',label='32 ms')
ax[0].errorbar(t, ep_mean64, yerr=ep_sem64,color='darkblue', linewidth=1, ecolor='lightsteelblue',label='64 ms')
ax[0].vlines(x=[1], ymin=(-6-0.5)*1e-6, ymax= (4+0.5)*1e-6, colors='black', ls='--')
ax[0].set_title('EEG Cap', loc='center', fontsize=10, pad=-10)

# ax[1].errorbar(t, ep_mean_subderm16, yerr=ep_sem_subderm16,color='green', linewidth=1, ecolor='lightgreen',label='16 ms')
# ax[1].errorbar(t, ep_mean_subderm32, yerr=ep_sem_subderm32,color='purple', linewidth=1, ecolor='thistle',label='32 ms')
# ax[1].errorbar(t, ep_mean_subderm64, yerr=ep_sem_subderm64,color='darkblue', linewidth=1, ecolor='lightsteelblue',label='64 ms')
# ax[1].vlines(x=[1], ymin=(-2-0.5)*1e-6, ymax= (4+0.5)*1e-6, colors='black', ls='--')
# ax[1].set_title('Subdermal (Vertex-Mastoid)', loc='center', fontsize=10)

ax[1].errorbar(t, -vertex16, yerr=-vertex_sem16,color='green', linewidth=1, ecolor='lightgreen',label='16 ms')
ax[1].errorbar(t,-vertex32, yerr=-vertex_sem32,color='purple', linewidth=1, ecolor='thistle',label='32 ms')
ax[1].errorbar(t, -vertex64, yerr=-vertex_sem64,color='darkblue', linewidth=1, ecolor='lightsteelblue',label='64 ms')
ax[1].vlines(x=[1], ymin=(-2-0.5)*1e-6, ymax= (4+0.5)*1e-6, colors='black', ls='--')
ax[1].set_title('Subdermal (Inverted Vertex)', loc='center', fontsize=10)

# ax[1].errorbar(t, mastoid16 -vertex16, yerr=mastoid_sem16-vertex_sem16,color='green', linewidth=1, ecolor='lightgreen',label='16 ms')
# ax[1].errorbar(t,mastoid32-vertex32, yerr=mastoid_sem32-vertex_sem32,color='purple', linewidth=1, ecolor='thistle',label='32 ms')
# ax[1].errorbar(t, mastoid64-vertex64, yerr=mastoid_sem64-vertex_sem64,color='darkblue', linewidth=1, ecolor='lightsteelblue',label='64 ms')
# ax[1].vlines(x=[1], ymin=(-6-0.5)*1e-6, ymax= (4+0.5)*1e-6, colors='black', ls='--')
# ax[1].set_title('Subdermal (Mastoid-Vertex)', loc='center', fontsize=10)

plt.xlim([-0.2, 2.1])
# plt.ylim([0.5,2])
ax[0].legend(prop={'size': 8})
plt.xlabel('Time (in seconds)')
# fig.text(0.0001, 0.5, 'ITC Value', va='center', rotation='vertical')
plt.suptitle('Q428 - Light Sedation')
plt.rcParams["figure.figsize"] = (6, 6)
plt.tight_layout()
plt.show()

# plt.savefig(save_loc + 'Q428_All.png', dpi=300)