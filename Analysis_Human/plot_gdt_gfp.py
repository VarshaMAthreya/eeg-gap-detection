# -*- coding: utf-8 -*-
"""
Created on Sun Feb  5 13:03:47 2023

@author: vmysorea
"""

import sys
sys.path.append('C:/Users/vmysorea/Documents/mne-python/')
import warnings
import mne
from matplotlib import pyplot as plt
from scipy.io import savemat
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
save_loc='C:/Users/vmysorea/Desktop/PhD/Stim_Analysis/GapDetection_EEG/Analysis/AnalyzedFiles_Figures/All_Ages/ITC_Figures/'
save_mat_loc = 'D:/PhD/Data/GDT_matfiles/'

subjlist_y = ['S273','S268','S269','S274','S282','S285','S259','S277','S279','S280','S270','S271','S281','S290','S284',
              'S303','S288','S260']

subjlist_o = ['S341','S312','S347','S340','S078','S069', 'S088','S072','S308','S344','S105','S291','S310','S339']

#S104 and S345 excluded (weird data) - S337 no EXGs
#%%Loading mat files for 20 condition only and plotting outside MNE for GFP 
evokeds_y = []
evokeds_o =[]
# evoked=a, evoked1 = b, evoked2 = c, evoked3 = c,fs=4096, t=epochs.times
gfps20_y = np.zeros((len(subjlist_y),11470))
evk_y = np.zeros((len(subjlist_y),11470))
evk_y1 = np.zeros((len(subjlist_y),11470))
evk_y2 = np.zeros((len(subjlist_y),11470))
evk_y3 = np.zeros((len(subjlist_y),11470))

for subj in range(len(subjlist_y)):
    sub = subjlist_y[subj]
    dat = io.loadmat(save_mat_loc + sub + '_GDTevokedall.mat', squeeze_me=True)
    dat.keys()
    evoked_y = dat['evoked']
    # evoked1_y = dat['evoked1']
    # evoked2_y = dat['evoked2']
    # evoked3_y = dat['evoked3']
    fs = dat['fs']
    t = dat['t']  
    gfp20_y=evoked_y.std(axis=0)   
    gfps20_y[subj,:]=gfp20_y
    evokeds_y = [evoked_y,]
    y = evoked_y.mean(axis=0)
    # y1 = evoked1_y.mean(axis=0)
    # y2 = evoked2_y.mean(axis=0)
    # y3 = evoked3_y.mean(axis=0)
    evk_y[subj,:] = y
    # evk_y1[subj,:] = y1
    # evk_y2[subj,:] = y2
    # evk_y3[subj,:] = y3
    
gfps20_o = np.zeros((len(subjlist_o),11470))
evk_o = np.zeros((len(subjlist_o),11470))
evk_o1 = np.zeros((len(subjlist_o),11470))
evk_o2 = np.zeros((len(subjlist_o),11470))
evk_o3 = np.zeros((len(subjlist_o),11470))

for subo in range(len(subjlist_o)):
    subs = subjlist_o[subo]
    dat = io.loadmat(save_mat_loc + subs + '_GDTevokedall.mat', squeeze_me=True)
    dat.keys()
    evoked_o = dat['evoked']
    # evoked1_o = dat['evoked1']
    # evoked2_o = dat['evoked2']
    # evoked3_o = dat['evoked3']
    fs = dat['fs']
    t = dat['t']   
    gfp20_o=evoked_o.std(axis=0)   
    gfps20_o[subo,:]=gfp20_o 
    evokeds_o += [evoked_o,]
    x = evoked_o.mean(axis=0)
    evk_o[subo,:] = x
    # x1 = evoked1_o.mean(axis=0)
    # x2 = evoked2_o.mean(axis=0)
    # x3 = evoked3_o.mean(axis=0)
    # evk_o1[subo,:] = x1
    # evk_o2[subo,:] = x2
    # evk_o3[subo,:] = x3
 
a = gfps20_y.mean(axis=0)*1e6
b = gfps20_o.mean(axis=0)*1e6
r = sem(gfps20_y)*1e6
s = sem(gfps20_o)*1e6
ymax=max(a)

# ##Plotting GFPs 
fig, ax = plt.subplots(constrained_layout=True)
plt.errorbar(t, a, yerr=r,  label = 'Below 35 y (N=21)', color='green', linewidth=2, ecolor='darkseagreen')
plt.errorbar(t, b, yerr=s, label = 'Above 35 y (N=15)', color='purple', linewidth=2, ecolor='thistle')
# plt.title('Binding across subjects - GFP')
plt.vlines(x=[0,0.5,1,1.5,2], ymin=0, ymax= ymax+0.15, colors='black', ls='--')
ax.text(0, ymax+0.15, 'Stim On', va='center', ha='center', fontsize = 12)
ax.text(0.5, ymax+0.15, 'Gap-16 ms', va='center', ha='center', fontsize = 11)
ax.text(1.0, ymax+0.15, 'Gap-32 ms', va='center', ha='center', fontsize = 11)
ax.text(1.5, ymax+0.15, 'Gap-64 ms', va='center', ha='center', fontsize = 11)
# ax.text(3.5, 0.5, 'Coherent', va='center', ha='center', fontsize = 11)
# ax.text(4.5, 0.5, 'Incoherent', va='center', ha='center', fontsize = 11)
ax.text(2, ymax+0.15, 'Stim End', va='center', ha='center', fontsize = 12)
# ax.axvspan(1.3,2, alpha=0.3,color='lightgrey')
# ax.axvspan(3.3,4, alpha=0.3,color='lightgrey')
plt.xlabel('Time(s)',fontsize=20)
plt.ylabel('Global Field Power(\u03bcV)',fontsize=20)
# plt.tight_layout()
plt.rcParams["figure.figsize"] = (7,5)
plt.ylim(0,ymax+0.13)
plt.xlim(-0.2,2.3)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(loc='upper right',fontsize='medium')
plt.show()

plt.savefig(save_loc + 'All_GDTGFP_2', dpi=500)

###Plotting evoked responses 
c = evk_y.mean(axis=0)*1e6
d = evk_o.mean(axis=0)*1e6
s=max(d)
e = sem(evk_y)*1e6
f = sem(evk_o)*1e6

##Plotting evokeds
fig1, ax = plt.subplots(constrained_layout=True)
plt.errorbar(t, c, yerr=e,  label = 'Below 35 y (N=19)', color='green', linewidth=2, ecolor='darkseagreen')
plt.errorbar(t, d, yerr=f, label = 'Above 35 y (N=14)', color='purple', linewidth=2, ecolor='thistle')
plt.vlines(x=[0.5,1,1.5,2], ymin=-0.3, ymax=0.7,colors='black', ls='--')
ax.text(0, s+0.25, 'Stim On', va='center', ha='center', fontsize = 12)
ax.text(0.5, s+0.25, 'Gap 16ms', va='center', ha='center', fontsize = 12)
ax.text(1, s+0.25, 'Gap 32ms', va='center', ha='center', fontsize = 12)
ax.text(1.5, s+0.25, 'Gap 64ms', va='center', ha='center', fontsize = 12)
ax.text(2, s+0.25, 'Stim End', va='center', ha='center', fontsize = 12)
plt.ylim(-0.3,0.7)
plt.xlim(0,2.2)
plt.xlabel('Time(s)')
plt.ylabel('Amplitude(\u03bcV)')
# plt.title('GDT across subjects - Evoked response')
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(loc='upper right',fontsize='medium')
plt.rcParams["figure.figsize"] = (6,5)
plt.xlabel('Time(s)',fontsize=16)
plt.ylabel('Amplitude(\u03bcV)',fontsize=16)
plt.show()



plt.savefig(save_loc + 'All_GDT_Evoked_Trial', dpi=300)