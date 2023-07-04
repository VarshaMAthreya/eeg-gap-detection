# -*- coding: utf-8 -*-
"""
Created on Thu May 25 21:10:02 2023

@author: vmysorea
"""
from scipy.stats import sem
import numpy as np
from scipy import io
from matplotlib import pyplot as plt
import sys
from scipy.io import savemat
import seaborn as sns
sys.path.append('C:/Users/vmysorea/mne-python/')
sys.path.append('C:/Users/vmysorea/ANLffr/')

# %% Plotting ITC for all subjects

# Loading data to plot ITC of ALL subjects - NOT ACROSS AGE

data_loc = ('D:/PhD/Data/EEG_GDT_Matfiles/')
save_loc = ('C:/Users/vmysorea/Desktop/PhD/Stim_Analysis/GapDetection_EEG/Analysis/AnalyzedFiles_Figures/All_Ages/ITC_Figures/')
subjlist = ['S273','S268','S269','S274','S282','S285','S277','S279','S280','S259','S270','S271','S281','S290'] 

evk16 = np.zeros((len(subjlist),2664))
evk32 = np.zeros((len(subjlist),2664))
evk64 = np.zeros((len(subjlist),2664))
onset16 = np.zeros((len(subjlist),10241))
onset32 = np.zeros((len(subjlist),10241))
onset64 = np.zeros((len(subjlist),10241))

for subj in range(len(subjlist)):
    sub = subjlist [subj]
    dat = io.loadmat(data_loc + sub + '_allevoked_A32_NB.mat', squeeze_me=True)
    dat.keys()
    onset_16 = dat['onset_16']
    onset_32 = dat['onset_32']
    onset_64 = dat['onset_64']
    evoked16 = dat['evoked_1']
    evoked32 = dat['evoked_2']
    evoked64 = dat['evoked_3']
    fs = dat['fs']
    t = dat['t'] 
    t1 = dat ['t1']
    evk16[subj,:] = evoked16
    evk32[subj,:] = evoked32
    evk64[subj,:] = evoked64
    onset16[subj,:] = onset_16
    onset32[subj,:] = onset_32
    onset64[subj,:] = onset_64

evk16_avg = evk16.mean(axis=0)*1e6
evk16_sem = sem(evk16)*1e6
evk32_avg = evk32.mean(axis=0)*1e6
evk32_sem = sem(evk32)*1e6
evk64_avg = evk64.mean(axis=0)*1e6
evk64_sem = sem(evk64)*1e6

onset16_avg = onset16.mean(axis=0)*1e6
onset16_sem = sem(onset16)*1e6
onset32_avg = onset32.mean(axis=0)*1e6
onset32_sem = sem(onset32)*1e6
onset64_avg = onset64.mean(axis=0)*1e6
onset64_sem = sem(onset64)*1e6

# Plot entire duration (Onsets) - All subjects
fig, ax = plt.subplots(3, 1, sharex=True, sharey=True, constrained_layout=True)
ax[0].errorbar(t, onset16_avg, yerr=onset16_sem,color='darkblue', linewidth=2, ecolor='lightsteelblue')
ax[0].set_title('Gap 16 ms', loc='center', fontsize=12)
ax[1].errorbar(t, onset32_avg, yerr=onset32_sem,color='purple', linewidth=2, ecolor='thistle')
ax[1].set_title('Gap 32 ms', loc='center', fontsize=12)
ax[2].errorbar(t, onset64_avg, yerr=onset64_sem,color='green', linewidth=2, ecolor='palegreen')
ax[2].set_title('Gap 64 ms', loc='center', fontsize=12)
# plt.xlim([-0.1, 1.1])
# plt.ylim([0.02, 0.09])
plt.xlabel('Time (in seconds)')
ax[1].set_ylabel('Amplitude(\u03bcV)')
#fig.text(-0.03,0.5, 'itc Value', va='center',rotation ='vertical')
fig.suptitle('Evoked Responses (N=' + str(len(subjlist)) + ')', fontsize=14)
params = {'legend.fontsize': 'x-large',
          'figure.figsize': (6, 5),
          'ytick.labelsize': 'xx-small'}
          #'ytick.major.pad': '6'}
plt.rcParams.update(params)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
#fig.supylabel('itc Value')
plt.show()

###Plotting - Option 2 
fig, ax = plt.subplots(constrained_layout=True)
plt.errorbar(t, onset16_avg, yerr=onset16_sem,color='darkblue', linewidth=2, ecolor='lightsteelblue', label = 'Gap 16 ms')
plt.errorbar(t, onset32_avg, yerr=onset32_sem,color='purple', linewidth=2, ecolor='thistle', label='Gap 32 ms')
plt.errorbar(t, onset64_avg, yerr=onset64_sem,color='green', linewidth=2, ecolor='palegreen', label='Gap 64 ms')
# plt.xlim([-0.1, 1.1])
# plt.ylim([0.02, 0.09])
plt.xlabel('Time (in seconds)')
plt.ylabel('Amplitude(\u03bcV)')
#fig.text(-0.03,0.5, 'itc Value', va='center',rotation ='vertical')
fig.suptitle('Evoked Responses (N=' + str(len(subjlist)) + ')', fontsize=14)
params = {'legend.fontsize': 'x-large',
          'figure.figsize': (6, 5),
          'ytick.labelsize': 'xx-small'}
          #'ytick.major.pad': '6'}
plt.rcParams.update(params)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.legend()
#fig.supylabel('itc Value')
plt.show()

# Plot gap durations combined - All subjects
fig, ax = plt.subplots(3, 1, sharex=True, sharey=True, constrained_layout=True)
ax[0].errorbar(t1, evk16_avg, yerr=evk16_sem,color='darkblue', linewidth=2, ecolor='lightsteelblue')
ax[0].set_title('Gap 16 ms', loc='center', fontsize=12)
ax[1].errorbar(t1, evk32_avg, yerr=evk32_sem,color='purple', linewidth=2, ecolor='thistle')
ax[1].set_title('Gap 32 ms', loc='center', fontsize=12)
ax[2].errorbar(t1, evk64_avg, yerr=evk64_sem,color='green', linewidth=2, ecolor='palegreen')
ax[2].set_title('Gap 64 ms', loc='center', fontsize=12)
# plt.xlim([-0.1, 1.1])
# plt.ylim([0.02, 0.09])
plt.xlabel('Time (in seconds)')
ax[1].set_ylabel('Amplitude(\u03bcV)')
#fig.text(-0.03,0.5, 'itc Value', va='center',rotation ='vertical')
fig.suptitle('Evoked responses for gaps (N=' + str(len(subjlist)) + ')', fontsize=14)
params = {'legend.fontsize': 'x-large',
          'figure.figsize': (6, 5),
          'ytick.labelsize': 'xx-small'}
          #'ytick.major.pad': '6'}
plt.rcParams.update(params)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
#fig.supylabel('itc Value')
plt.show()

###Plotting - Option 2 
from matplotlib.colors import ListedColormap
cmap = ListedColormap(sns.color_palette())

palette = sns.color_palette("rocket_r", as_cmap=True)

fig, ax = plt.subplots(constrained_layout=True)
plt.errorbar(t1, evk16_avg, yerr=evk16_sem,color='darkblue', linewidth=2, ecolor='lightsteelblue',alpha=0.5,label='Gap 16 ms')
plt.errorbar(t1, evk32_avg, yerr=evk32_sem,color='purple', linewidth=2, ecolor='thistle',alpha=0.5, label='Gap 32 ms')
plt.errorbar(t1, evk64_avg, yerr=evk64_sem,color='green', linewidth=2, ecolor='palegreen',alpha=0.5, label='Gap 64 ms')
# plt.xlim([-0.1, 1.1])
# plt.ylim([0.02, 0.09])
plt.xlabel('Time (in seconds)')
plt.ylabel('Amplitude(\u03bcV)')
#fig.text(-0.03,0.5, 'itc Value', va='center',rotation ='vertical')
fig.suptitle('Evoked Responses (N=' + str(len(subjlist)) + ')', fontsize=14)
params = {'legend.fontsize': 'x-large',
          'figure.figsize': (6, 5),
          'ytick.labelsize': 'xx-small'}
          #'ytick.major.pad': '6'}
plt.rcParams.update(params)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.legend()
#fig.supylabel('itc Value')
plt.show()

####Option 3
t2 = t[t<0.55]
onset16_500 = onset16_avg[t<0.55]
onset16_sem_500 = onset16_sem[t<0.55]
onset32_500 = onset32_avg[t<0.55]
onset32_sem_500 = onset32_sem[t<0.55]
onset64_500 = onset64_avg[t<0.55]
onset64_sem_500 = onset64_sem[t<0.55]

plt.errorbar(t2, onset16_500, yerr=onset16_sem_500,color='darkblue', linewidth=2, ecolor='lightsteelblue',alpha=0.5,label='Gap 16 ms')
plt.errorbar(t2, onset32_500, yerr=onset32_sem_500,color='purple', linewidth=2, ecolor='thistle',alpha=0.5, label='Gap 32 ms')
plt.errorbar(t2, onset64_500, yerr=onset64_sem_500,color='green', linewidth=2, ecolor='palegreen',alpha=0.5, label='Gap 64 ms')
plt.legend()
plt.show()


t3=np.arange(0.5,1.05,1/fs)
evk16_500 = evk16_avg[t1>0]
evk16_sem_500 = evk16_sem[t1>0]
evk32_500 = evk32_avg[t1>0]
evk32_sem_500 = evk32_sem[t1>0]
evk64_500 = evk64_avg[t1>0]
evk64_sem_500 = evk64_sem[t1>0]

fig, ax = plt.subplots(3, 1, sharex=True, sharey=True, constrained_layout=True)
ax[0].errorbar(t2, onset16_500, yerr=onset16_sem_500,color='darkblue', linewidth=2, ecolor='lightsteelblue',alpha=0.5)
ax[0].errorbar(t3,evk16_500, yerr=evk16_sem_500,color='darkblue', linewidth=2, ecolor='lightsteelblue',alpha=0.5)
ax[0].set_title('Gap 16 ms', loc='center', fontsize=12)
ax[1].errorbar(t2, onset32_500, yerr=onset32_sem_500,color='purple', linewidth=2, ecolor='thistle',alpha=0.5)
ax[1].errorbar(t3, evk32_500, yerr=evk32_sem_500,color='purple', linewidth=2, ecolor='thistle',alpha=0.5)
ax[1].set_title('Gap 32 ms', loc='center', fontsize=12)
ax[2].errorbar(t2, onset64_500, yerr=onset64_sem_500,color='green', linewidth=2, ecolor='palegreen',alpha=0.5)
ax[2].errorbar(t3, evk64_500, yerr=evk64_sem_500,color='green', linewidth=2, ecolor='palegreen',alpha=0.5)
ax[2].set_title('Gap 64 ms', loc='center', fontsize=12)
# plt.xlim([-0.1, 1.1])
# plt.ylim([0.02, 0.09])
plt.xlabel('Time (in seconds)')
ax[1].set_ylabel('Amplitude(\u03bcV)')
#fig.text(-0.03,0.5, 'itc Value', va='center',rotation ='vertical')
fig.suptitle('Evoked Responses (N=' + str(len(subjlist)) + ')', fontsize=14)
params = {'legend.fontsize': 'x-large',
          'figure.figsize': (6, 5),
          'ytick.labelsize': 'xx-small'}
          #'ytick.major.pad': '6'}
plt.rcParams.update(params)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
#fig.supylabel('itc Value')
plt.show()