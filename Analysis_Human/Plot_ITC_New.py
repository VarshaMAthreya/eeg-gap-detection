# -*- coding: utf-8 -*-
"""
Created on Wed May 24 23:44:56 2023

@author: vmysorea
"""
from scipy.stats import sem
import numpy as np
from scipy import io
from matplotlib import pyplot as plt
import sys
from scipy.io import savemat
sys.path.append('C:/Users/vmysorea/mne-python/')
sys.path.append('C:/Users/vmysorea/ANLffr/')

# %% Plotting ITC for all subjects

# Loading data to plot ITC of ALL subjects - NOT ACROSS AGE

data_loc = ('D:/PhD/Data/EEG_GDT_Matfiles/')
save_loc = ('C:/Users/vmysorea/Desktop/PhD/Stim_Analysis/GapDetection_EEG/Analysis/AnalyzedFiles_Figures/All_Ages/ITC_Figures/')
Subjects = ['S273','S268','S269','S274','S282','S285','S277','S279','S280','S259','S270','S271','S281','S290'] 

itc1_mean = np.zeros((len(Subjects), 2664))
itc2_mean = np.zeros((len(Subjects), 2664))
itc3_mean = np.zeros((len(Subjects), 2664))
itc1_300ms_avg = np.zeros((len(Subjects), 1229))

for sub in range(len(Subjects)):
    subj = Subjects[sub]
    dat = io.loadmat(data_loc + subj + '_ITC_A32.mat', squeeze_me=True)

    dat.keys()
    itc1 = dat['itc1']
    itc2 = dat['itc2']
    itc3 = dat['itc3']
    freqs = dat['freqs']
    n_channels=dat['n_channels']
    t = dat['t']
    itc1_avg = itc1[:, :].mean(axis=0)
    itc1_mean[sub, :] = itc1_avg
    itc2_avg = itc2[:, :].mean(axis=0)
    itc2_mean[sub, :] = itc2_avg
    itc3_avg = itc3[:, :].mean(axis=0)
    itc3_mean[sub, :] = itc3_avg

itc1_all = itc1_mean.mean(axis=0)
itc1_sem = sem(itc1_mean)
itc2_all = itc2_mean.mean(axis=0)
itc2_sem = sem(itc2_mean)
itc3_all = itc3_mean.mean(axis=0)
itc3_sem = sem(itc3_mean)

# Subplots - All subjects
fig, ax = plt.subplots(3, 1, sharex=True, sharey=True, constrained_layout=True)
ax[0].errorbar(t, itc1_all, yerr=itc1_sem,color='darkblue', linewidth=2, ecolor='lightsteelblue')
ax[0].set_title('ITC - Gap 16 ms', loc='center', fontsize=12)
ax[1].errorbar(t, itc2_all, yerr=itc2_sem,color='purple', linewidth=2, ecolor='thistle')
ax[1].set_title('ITC - Gap 32 ms', loc='center', fontsize=12)
ax[2].errorbar(t, itc3_all, yerr=itc3_sem,color='green', linewidth=2, ecolor='palegreen')
ax[2].set_title('ITC - Gap 64 ms', loc='center', fontsize=12)
# plt.xlim([-0.1, 1.1])
# plt.ylim([0.02, 0.09])
plt.xlabel('Time (in seconds)')
ax[1].set_ylabel('ITC Value')
#fig.text(-0.03,0.5, 'itc Value', va='center',rotation ='vertical')
fig.suptitle('ITC - All Subjects (N=' + str(len(Subjects)) + ')', fontsize=14)
params = {'legend.fontsize': 'x-large',
          'figure.figsize': (8, 5),
          'ytick.labelsize': 'xx-small'}
          #'ytick.major.pad': '6'}
plt.rcParams.update(params)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
#fig.supylabel('itc Value')
plt.show()
#x=0.55,
# plt.savefig(save_loc + 'ITC123_Combined_AllAges.png', dpi=300)
