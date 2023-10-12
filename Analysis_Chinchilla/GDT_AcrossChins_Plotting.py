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
import seaborn as sns

plt.switch_backend('QT5Agg')  # Making the plots interactive (Scrollable)
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')
# Defining the dimensions and quality of figures
plt.rcParams["figure.figsize"] = (5.5,5)
plt.rcParams['figure.dpi'] = 120
#%%Setting up stuff
fig_loc = 'C:/Users/vmysorea/Desktop/PhD/Stim_Analysis/MTB_Analysis/GreenLight/'
save_loc = 'D:/PhD/Data/Chin_Data/AnalyzedGDT_matfiles/'

subjlist_ynh = ['Q410', 'Q422'] 
# subjlist_mnh = ['Q351']
# subjlist_tts = ['Q410']

#%%Loading mat files -- Need to work on this later

for subj in range(len(subjlist_ynh)):
    sub = subjlist_ynh[subj]
    dat1 = io.loadmat(save_loc + sub + '_16ms.mat', squeeze_me=True)
    dat2 = io.loadmat(save_loc + sub + '_32ms.mat', squeeze_me=True)
    dat3 = io.loadmat(save_loc + sub + '_64ms.mat', squeeze_me=True)

    dat1.keys()
    dat2.keys()
    dat3.keys()
    
    ynh_ep_mastoid16 = dat1['ep_mastoid']
    ynh_ep_vertex16 = dat1['ep_vertex']
    ynh_ep_ground16= dat1['ep_ground']
    ynh_ep_all16 = dat1['ep_all']
    ynh_ep_mean16 = (dat1['ep_mean'])*1e6
    ynh_ep_sem16 = (dat1['ep_sem'])*1e6
    ynh_ep_subderm16 = dat1['ep_subderm']
    ynh_ep_mean_subderm16 = dat1['ep_mean_subderm']
    ynh_ep_sem_subderm16 = dat1['ep_sem_subderm']
    picks= dat1['picks']
    t=dat1['t']
    
    ynh_ep_mastoid32 = dat2['ep_mastoid']
    ynh_ep_vertex32 = dat2['ep_vertex']
    ynh_ep_ground32= dat2['ep_ground']
    ynh_ep_all32 = dat2['ep_all']
    ynh_ep_mean32 = (dat2['ep_mean'])*1e6
    ynh_ep_sem32 = (dat2['ep_sem'])*1e6
    ynh_ep_subderm32 = dat2['ep_subderm']
    ynh_ep_mean_subderm32 = dat2['ep_mean_subderm']
    ynh_ep_sem_subderm32 = dat2['ep_sem_subderm']
    
    ynh_ep_mastoid64 = dat3['ep_mastoid']
    ynh_ep_vertex64 = dat3['ep_vertex']
    ynh_ep_ground64= dat3['ep_ground']
    ynh_ep_all64 = dat3['ep_all']
    ynh_ep_mean64 = (dat3['ep_mean'])*1e6
    ynh_ep_sem64 = (dat3['ep_sem'])*1e6
    ynh_ep_subderm64 = dat3['ep_subderm']
    ynh_ep_mean_subderm64 = dat3['ep_mean_subderm']
    ynh_ep_sem_subderm64 = dat3['ep_sem_subderm']
    
ynh_vertex16 = ynh_ep_vertex16.mean(axis=0)
ynh_vertex_sem16 = ynh_ep_vertex16.std(axis=0) / np.sqrt(ynh_ep_vertex16.shape[0])
ynh_mastoid16 = ynh_ep_mastoid16.mean(axis=0)
ynh_mastoid_sem16 = ynh_ep_mastoid16.std(axis=0) / np.sqrt(ynh_ep_mastoid16.shape[0])

ynh_vertex32 = ynh_ep_vertex32.mean(axis=0)
ynh_vertex_sem32 = ynh_ep_vertex32.std(axis=0) / np.sqrt(ynh_ep_vertex32.shape[0])
ynh_mastoid32 = ynh_ep_mastoid32.mean(axis=0)
ynh_mastoid_sem32 = ynh_ep_mastoid32.std(axis=0) / np.sqrt(ynh_ep_mastoid32.shape[0])

ynh_vertex64 = ynh_ep_vertex64.mean(axis=0)
ynh_vertex_sem64 = ynh_ep_vertex64.std(axis=0) / np.sqrt(ynh_ep_vertex64.shape[0])
ynh_mastoid64 = ynh_ep_mastoid64.mean(axis=0)
ynh_mastoid_sem64 = ynh_ep_mastoid64.std(axis=0) / np.sqrt(ynh_ep_mastoid64.shape[0])


#%%##MNH 

# for subj in range(len(subjlist_mnh)):
#     sub = subjlist_mnh[subj]
#     dat1 = io.loadmat(save_loc + sub + '_16ms.mat', squeeze_me=True)
#     dat2 = io.loadmat(save_loc + sub + '_32ms.mat', squeeze_me=True)
#     dat3 = io.loadmat(save_loc + sub + '_64ms.mat', squeeze_me=True)

#     dat1.keys()
#     dat2.keys()
#     dat3.keys()
    
#     mnh_ep_mastoid16 = dat1['ep_mastoid']
#     mnh_ep_vertex16 = dat1['ep_vertex']
#     mnh_ep_ground16= dat1['ep_ground']
#     mnh_ep_all16 = dat1['ep_all']
#     mnh_ep_mean16 = dat1['ep_mean']
#     mnh_ep_sem16 = dat1['ep_sem']
#     mnh_ep_subderm16 = dat1['ep_subderm']
#     mnh_ep_mean_subderm16 = dat1['ep_mean_subderm']
#     mnh_ep_sem_subderm16 = dat1['ep_sem_subderm']
#     picks= dat1['picks']
#     t=dat1['t']
    
#     mnh_ep_mastoid32 = dat2['ep_mastoid']
#     mnh_ep_vertex32 = dat2['ep_vertex']
#     mnh_ep_ground32= dat2['ep_ground']
#     mnh_ep_all32 = dat2['ep_all']
#     mnh_ep_mean32 = dat2['ep_mean']
#     mnh_ep_sem32 = dat2['ep_sem']
#     mnh_ep_subderm32 = dat2['ep_subderm']
#     mnh_ep_mean_subderm32 = dat2['ep_mean_subderm']
#     mnh_ep_sem_subderm32 = dat2['ep_sem_subderm']
    
#     mnh_ep_mastoid64 = dat3['ep_mastoid']
#     mnh_ep_vertex64 = dat3['ep_vertex']
#     mnh_ep_ground64= dat3['ep_ground']
#     mnh_ep_all64 = dat3['ep_all']
#     mnh_ep_mean64 = dat3['ep_mean']
#     mnh_ep_sem64 = dat3['ep_sem']
#     mnh_ep_subderm64 = dat3['ep_subderm']
#     mnh_ep_mean_subderm64 = dat3['ep_mean_subderm']
#     mnh_ep_sem_subderm64 = dat3['ep_sem_subderm']
    
# mnh_vertex16 = mnh_ep_vertex16.mean(axis=0)
# mnh_vertex_sem16 = mnh_ep_vertex16.std(axis=0) / np.sqrt(mnh_ep_vertex16.shape[0])
# mnh_mastoid16 = mnh_ep_mastoid16.mean(axis=0)
# mnh_mastoid_sem16 = mnh_ep_mastoid16.std(axis=0) / np.sqrt(mnh_ep_mastoid16.shape[0])

# mnh_vertex32 = mnh_ep_vertex32.mean(axis=0)
# mnh_vertex_sem32 = mnh_ep_vertex32.std(axis=0) / np.sqrt(mnh_ep_vertex32.shape[0])
# mnh_mastoid32 = mnh_ep_mastoid32.mean(axis=0)
# mnh_mastoid_sem32 = mnh_ep_mastoid32.std(axis=0) / np.sqrt(mnh_ep_mastoid32.shape[0])

# mnh_vertex64 = mnh_ep_vertex64.mean(axis=0)
# mnh_vertex_sem64 = mnh_ep_vertex64.std(axis=0) / np.sqrt(mnh_ep_vertex64.shape[0])
# mnh_mastoid64 = mnh_ep_mastoid64.mean(axis=0)
# mnh_mastoid_sem64 = mnh_ep_mastoid64.std(axis=0) / np.sqrt(mnh_ep_mastoid64.shape[0])


#%%##TTS

# for subj in range(len(subjlist_tts)):
#     sub = subjlist_tts[subj]
#     dat1 = io.loadmat(save_loc + sub + '_16ms.mat', squeeze_me=True)
#     dat2 = io.loadmat(save_loc + sub + '_32ms.mat', squeeze_me=True)
#     dat3 = io.loadmat(save_loc + sub + '_64ms.mat', squeeze_me=True)

#     dat1.keys()
#     dat2.keys()
#     dat3.keys()
    
#     tts_ep_mastoid16 = dat1['ep_mastoid']
#     tts_ep_vertex16 = dat1['ep_vertex']
#     tts_ep_ground16= dat1['ep_ground']
#     tts_ep_all16 = dat1['ep_all']
#     tts_ep_mean16 = dat1['ep_mean']
#     tts_ep_sem16 = dat1['ep_sem']
#     tts_ep_subderm16 = dat1['ep_subderm']
#     tts_ep_mean_subderm16 = dat1['ep_mean_subderm']
#     tts_ep_sem_subderm16 = dat1['ep_sem_subderm']
#     picks= dat1['picks']
#     t=dat1['t']
    
#     tts_ep_mastoid32 = dat2['ep_mastoid']
#     tts_ep_vertex32 = dat2['ep_vertex']
#     tts_ep_ground32= dat2['ep_ground']
#     tts_ep_all32 = dat2['ep_all']
#     tts_ep_mean32 = dat2['ep_mean']
#     tts_ep_sem32 = dat2['ep_sem']
#     tts_ep_subderm32 = dat2['ep_subderm']
#     tts_ep_mean_subderm32 = dat2['ep_mean_subderm']
#     tts_ep_sem_subderm32 = dat2['ep_sem_subderm']
    
#     tts_ep_mastoid64 = dat3['ep_mastoid']
#     tts_ep_vertex64 = dat3['ep_vertex']
#     tts_ep_ground64= dat3['ep_ground']
#     tts_ep_all64 = dat3['ep_all']
#     tts_ep_mean64 = dat3['ep_mean']
#     tts_ep_sem64 = dat3['ep_sem']
#     tts_ep_subderm64 = dat3['ep_subderm']
#     tts_ep_mean_subderm64 = dat3['ep_mean_subderm']
#     tts_ep_sem_subderm64 = dat3['ep_sem_subderm']
    
# tts_vertex16 = tts_ep_vertex16.mean(axis=0)
# tts_vertex_sem16 = tts_ep_vertex16.std(axis=0) / np.sqrt(tts_ep_vertex16.shape[0])
# tts_mastoid16 = tts_ep_mastoid16.mean(axis=0)
# tts_mastoid_sem16 = tts_ep_mastoid16.std(axis=0) / np.sqrt(tts_ep_mastoid16.shape[0])

# tts_vertex32 = tts_ep_vertex32.mean(axis=0)
# tts_vertex_sem32 = tts_ep_vertex32.std(axis=0) / np.sqrt(tts_ep_vertex32.shape[0])
# tts_mastoid32 = tts_ep_mastoid32.mean(axis=0)
# tts_mastoid_sem32 = tts_ep_mastoid32.std(axis=0) / np.sqrt(tts_ep_mastoid32.shape[0])

# tts_vertex64 = tts_ep_vertex64.mean(axis=0)
# tts_vertex_sem64 = tts_ep_vertex64.std(axis=0) / np.sqrt(tts_ep_vertex64.shape[0])
# tts_mastoid64 = tts_ep_mastoid64.mean(axis=0)
# tts_mastoid_sem64 = tts_ep_mastoid64.std(axis=0) / np.sqrt(tts_ep_mastoid64.shape[0])



#%%##Plotting 

sns.set_palette ("Dark2")

fig, ax = plt.subplots(1, 1, sharex=True, sharey=True)
ax.plot(t,  ynh_ep_mean16, label='16 ms', alpha=0.9)
ax.fill_between(t,  ynh_ep_mean16 - ynh_ep_sem16,  ynh_ep_mean16 + ynh_ep_sem16, alpha=0.3)
ax.plot(t,  ynh_ep_mean32, label='32 ms', alpha=0.9)
ax.fill_between(t,  ynh_ep_mean32 - ynh_ep_sem32,  ynh_ep_mean32 + ynh_ep_sem32, alpha=0.3)
ax.plot(t,  ynh_ep_mean64, label='64 ms', alpha=0.9)
ax.fill_between(t,  ynh_ep_mean64 - ynh_ep_sem64,  ynh_ep_mean64 + ynh_ep_sem64, alpha=0.3)

y_limits = ax.get_ylim()
ax.axvline(x=0, color = 'black',linestyle='--', alpha=0.7)
ax.text(0, y_limits[1] + 0.01, 'Stim On', ha='center', weight='bold')
ax.axvline(x=1, color='blue', linestyle='--', alpha=0.7)
ax.text(1, y_limits[1] + 0.01, 'Gap', ha='center')

plt.xlim([-0.2, 2.1])
# plt.ylim([0.5,2])
ax.legend(prop={'size': 8})
plt.xlabel('Time (in seconds)',fontsize=12)
fig.text(0.0001, 0.5, 'Amplitude (\u03bcV)', va='center', rotation='vertical',fontsize=12)
plt.suptitle('Cortical GDT | Light Sedation (N=3)', fontsize=14)
plt.rcParams["figure.figsize"] = (6.5, 6)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

plt.savefig(fig_loc + 'ChinGDT_AcrossGroups_1.png', dpi=400)

# fig, ax = plt.subplots(1, 1, sharex=True, sharey=True)
# ax.errorbar(t, ynh_ep_mean16*1e6, yerr=ynh_ep_sem16*1e6, linewidth=1,label='16 ms' +str(picks))
# ax.errorbar(t, ynh_ep_mean32*1e6, yerr=ynh_ep_sem32*1e6, linewidth=1, label='32 ms')
# ax.errorbar(t, ynh_ep_mean64*1e6, yerr=ynh_ep_sem64*1e6,linewidth=1, label='64 ms')
# ax.vlines(x=[1], ymin=(-6-0.5), ymax= (4+0.5), colors='black', ls='--')
# ax.set_title('YNH - EEG Cap', loc='center', fontsize=12)

# ax[1].errorbar(t, mnh_ep_mean16*1e6, yerr=mnh_ep_sem16*1e6,color='green', linewidth=1, ecolor='lightgreen',label='16 ms' +str(picks))
# ax[1].errorbar(t, mnh_ep_mean32*1e6, yerr=mnh_ep_sem32*1e6,color='purple', linewidth=1, ecolor='thistle',label='32 ms')
# ax[1].errorbar(t, mnh_ep_mean64*1e6, yerr=mnh_ep_sem64*1e6,color='darkblue', linewidth=1, ecolor='lightsteelblue',label='64 ms')
# ax[1].vlines(x=[1], ymin=(-6-0.5), ymax= (4+0.5), colors='black', ls='--')
# ax[1].set_title('MNH - EEG Cap', loc='center', fontsize=12)

# ax[2].errorbar(t, tts_ep_mean16*1e6, yerr=tts_ep_sem16*1e6,color='green', linewidth=1, ecolor='lightgreen',label='16 ms' +str(picks))
# ax[2].errorbar(t, tts_ep_mean32*1e6, yerr=tts_ep_sem32*1e6,color='purple', linewidth=1, ecolor='thistle',label='32 ms')
# ax[2].errorbar(t, tts_ep_mean64*1e6, yerr=tts_ep_sem64*1e6,color='darkblue', linewidth=1, ecolor='lightsteelblue',label='64 ms')
# ax[2].vlines(x=[1], ymin=(-6-0.5), ymax= (4+0.5), colors='black', ls='--')
# ax[2].set_title('TTS - EEG Cap', loc='center', fontsize=12)

# ax[1].errorbar(t, ep_mean_subderm16, yerr=ep_sem_subderm16,color='green', linewidth=1, ecolor='lightgreen',label='16 ms')
# ax[1].errorbar(t, ep_mean_subderm32, yerr=ep_sem_subderm32,color='purple', linewidth=1, ecolor='thistle',label='32 ms')
# ax[1].errorbar(t, ep_mean_subderm64, yerr=ep_sem_subderm64,color='darkblue', linewidth=1, ecolor='lightsteelblue',label='64 ms')
# ax[1].vlines(x=[1], ymin=(-2-0.5)*1e-6, ymax= (4+0.5)*1e-6, colors='black', ls='--')
# ax[1].set_title('Subdermal (Vertex-Mastoid)', loc='center', fontsize=10)

# ax[1].errorbar(t, -vertex16, yerr=-vertex_sem16,color='green', linewidth=1, ecolor='lightgreen',label='16 ms')
# ax[1].errorbar(t,-vertex32, yerr=-vertex_sem32,color='purple', linewidth=1, ecolor='thistle',label='32 ms')
# ax[1].errorbar(t, -vertex64, yerr=-vertex_sem64,color='darkblue', linewidth=1, ecolor='lightsteelblue',label='64 ms')
# ax[1].vlines(x=[1], ymin=(-2-0.5)*1e-6, ymax= (4+0.5)*1e-6, colors='black', ls='--')
# ax[1].set_title('Subdermal (Inverted Vertex)', loc='center', fontsize=10)

# ax[1].errorbar(t, mastoid16 -vertex16, yerr=mastoid_sem16-vertex_sem16,color='green', linewidth=1, ecolor='lightgreen',label='16 ms')
# ax[1].errorbar(t,mastoid32-vertex32, yerr=mastoid_sem32-vertex_sem32,color='purple', linewidth=1, ecolor='thistle',label='32 ms')
# ax[1].errorbar(t, mastoid64-vertex64, yerr=mastoid_sem64-vertex_sem64,color='darkblue', linewidth=1, ecolor='lightsteelblue',label='64 ms')
# ax[1].vlines(x=[1], ymin=(-6-0.5)*1e-6, ymax= (4+0.5)*1e-6, colors='black', ls='--')
# ax[1].set_title('Subdermal (Mastoid-Vertex)', loc='center', fontsize=10)

# plt.xlim([-0.2, 2.1])
# # plt.ylim([0.5,2])
# ax.legend(prop={'size': 8})
# plt.xlabel('Time (in seconds)',fontsize=12)
# fig.text(0.0001, 0.5, 'Amplitude (\u03bcV)', va='center', rotation='vertical',fontsize=12)
# plt.suptitle('Cortical GDT | Light Sedation', fontsize=14)
# plt.rcParams["figure.figsize"] = (6.5, 6)
# plt.tight_layout(rect=[0, 0.03, 1, 0.95])
# plt.show()

