# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 10:30:59 2023

@author: vmysorea
"""

import sys
sys.path.append('C:/Users/vmysorea/Documents/mne-python/')
import warnings
from matplotlib import pyplot as plt
from scipy import io
import numpy as np
import scipy.stats as stats
import seaborn as sns
import pandas as pd


plt.switch_backend('QT5Agg')  # Making the plots interactive (Scrollable)
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')
# Defining the dimensions and quality of figures
plt.rcParams["figure.figsize"] = (5.5,5)
plt.rcParams['figure.dpi'] = 120

# %%Setting up stuff
fig_loc = 'C:/Users/vmysorea/Desktop/PhD/Conferences/ARO 2024/Chin_miniEEG/Figures/'
data_loc = 'D:/PhD/Data/Chin_Data/AnalyzedGDT_matfiles/'
csv_loc = 'C:/Users/vmysorea/Desktop/PhD/Stim_Analysis/MTB_Analysis/'

subjlist = ['Q419']

for subj in range(len(subjlist)):
    sub = subjlist[subj]
    dat1 = io.loadmat(data_loc + sub + '_Anesthetized_64ms_2-20Hz.mat', squeeze_me=True)
    dat1.keys()
    t = dat1['t']  # For GDT and ITC
    t_full = dat1['t_full']
    cap_all = dat1['ep_all'][0:6]
    gap_all = dat1['gap_cap'][0:6]

cap = cap_all.mean(axis=0)*1e6
gap = gap_all.mean(axis=0)*1e6
cap_sem = stats.sem(cap_all)*1e6
gap_sem = stats.sem(gap_all)*1e6

# plt.errorbar(t_full, cap, yerr=cap_sem)
# plt.show()

subjlist_ls = ['Q412']

# subjlist_ls = ['Q412', 'Q422', 'Q424', 'Q426', 'Q428',
#                'Q351', 'Q363', 'Q364', 'Q365', 'Q368',
#                'Q402', 'Q404', 'Q406', 'Q407', 'Q410']

caps_ls = np.zeros((1, 10241))
gaps_ls = np.zeros((1, 5735))
caps_ls_all = []
gaps_ls_all = []


for subj in range(len(subjlist_ls)):
    sub = subjlist_ls[subj]
    dat = io.loadmat(data_loc + sub + '_64ms_2-20Hz.mat', squeeze_me=True)
    dat.keys()
    t = dat['t']  # For GDT and ITC
    t_full = dat['t_full']
    cap_ls = (dat['ep_all'][0:6]).mean(axis=0)
    gap_ls = (dat['gap_cap'][0:6]).mean(axis=0)
    caps_ls[subj,:]=+ cap_ls
    # caps_ls_all += [caps_ls,]
    gaps_ls[subj,:]=+ gap_ls 
    # gaps_ls_all += [gaps_ls,]
    
cap1 = caps_ls.mean(axis=0)*1e6
gap1 = gaps_ls.mean(axis=0)*1e6
cap_sem1 = stats.sem(caps_ls)*1e6
gap_sem1 = stats.sem(gaps_ls)*1e6

fig, ax = plt.subplots(figsize=(9,7),constrained_layout=True)
plt.errorbar(t_full, cap1, yerr=cap_sem1, label = 'Light Sedation (N=1)', color='green', linewidth=4, ecolor='palegreen')
plt.errorbar(t_full, cap, yerr=cap_sem, label = 'Anesthetized (N=1)', color='purple', linewidth=4, ecolor='thistle')
# ax.set_title('Cortical Onset Response', pad=15, fontsize = 26, weight = 'bold')
plt.vlines(x=(0),ymin=-3.5, ymax=3, color='black', linestyle='--', alpha=1,linewidth=3)
# plt.vlines(x=1, ymin=-3.5, ymax=3,color='blue', linestyle='--', alpha=1)
ax.text(0, 3.1, 'Stim On', va='center', ha='center', fontsize = 19, weight='bold')
# ax.text(1, 3.1, 'Gap', va='center', ha='center', fontsize = 11, weight='bold')
# ax.text(2, 3.1, 'Stim End', va='center', ha='center', fontsize = 12, weight='bold')
# ax.fill_between(x=[0,0.35], y1=-3.5, y2=3, color='gray', alpha=0.3)
plt.xlabel('Time(s)',fontsize=26, weight = 'bold')
plt.ylabel('Amplitude(\u03bcV)',fontsize=26, weight = 'bold')
plt.rcParams["figure.figsize"] = (6.5,5)
plt.ylim(-3.5,3)
plt.xlim(-0.2,0.6)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.legend(loc='upper right',fontsize='xx-large')
plt.show()

plt.savefig(fig_loc + "GDT_LSvsSed_SelChan_Onset.png", dpi=500, bbox_inches="tight", transparent=True)