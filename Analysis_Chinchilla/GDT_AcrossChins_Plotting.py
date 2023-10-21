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
fig_loc = 'C:/Users/vmysorea/Desktop/PhD/Stim_Analysis/MTB_Analysis/FinalThesis/'
data_loc = 'D:/PhD/Data/Chin_Data/AnalyzedGDT_matfiles/'
csv_loc = 'C:/Users/vmysorea/Desktop/PhD/Stim_Analysis/MTB_Analysis/'

subjlist = ['Q351', 'Q363', 'Q364', 'Q365', 'Q368',
            'Q402', 'Q404', 'Q406', 'Q407', 'Q410',
            'Q412', 'Q422', 'Q424', 'Q426', 'Q428'] 

# Load data with subjects' age and gender and pick subj only from subjlist
dat = pd.read_csv(csv_loc + 'chin_group_age.csv')
dat0 = dat['Subject'].isin(subjlist)
dat = dat[dat0]
        
#Loading mat files for each subject and loading all the variables 
data_dict = {}
for index, row in dat.iterrows():
    subj = row['Subject']
    mat_data = io.loadmat(data_loc + subj + '_AllGaps_2-20Hz.mat', squeeze_me=True)
    
    gapcap16_sel = mat_data['gap_cap16'][3:7]
    gapcap32_sel = mat_data['gap_cap32'][3:7]
    gapcap64_sel = mat_data['gap_cap64'][3:7]
    
    cap16_sel = mat_data['cap16'][0:3]
    cap32_sel = mat_data['cap32'][0:3]
    cap64_sel = mat_data['cap64'][0:3]

    data_dict[subj] = { 'picks' : mat_data['picks'],
                        'gapcap16' : (mat_data['gap_cap16']).mean(axis=0),
                        'gapcap32' : (mat_data['gap_cap32']).mean(axis=0),
                        'gapcap64' : (mat_data['gap_cap64']).mean(axis=0),
                        'gapground16' : mat_data['gap_ground16'],
                        'gapground32' : mat_data['gap_ground32'],
                        'gapground64' : mat_data['gap_ground64'],
                        'gapmastoid16' : mat_data['gap_mastoid16'],
                        'gapmastoid32' : mat_data['gap_mastoid32'],
                        'gapmastoid64' : mat_data['gap_mastoid64'],
                        'gapvertex16' : mat_data['gap_vertex16'],
                        'gapvertex32' : mat_data['gap_vertex32'],
                        'gapvertex64' : mat_data['gap_vertex64'],
                        'cap16' : (mat_data['cap16']).mean(axis=0),
                        'cap32' : (mat_data['cap32']).mean(axis=0),
                        'cap64' : (mat_data['cap64']).mean(axis=0),
                        'ground16' : mat_data['ground16'],
                        'ground32' : mat_data['ground32'],
                        'ground64' : mat_data['ground64'],
                        'mastoid16' : mat_data['mastoid16'],
                        'mastoid32' : mat_data['mastoid32'],
                        'mastoid64' : mat_data['mastoid64'],
                        'vertex16' : mat_data['vertex16'],
                        'vertex32' : mat_data['vertex32'],
                        'vertex64' : mat_data['vertex64'],
                        't' : mat_data['t'],
                        't_full' : mat_data['t_full'],
                        'gapcap16_sel' : (gapcap16_sel).mean(axis=0),
                        'gapcap32_sel' : (gapcap32_sel).mean(axis=0),
                        'gapcap64_sel' : (gapcap64_sel).mean(axis=0),
                        'cap16_sel' : (cap16_sel).mean(axis=0),
                        'cap32_sel' : (cap32_sel).mean(axis=0),
                        'cap64_sel' : (cap64_sel).mean(axis=0)}
                              
##Organizing mat data by groups 
grouped_data = {}
for index, row in dat.iterrows():
    subj = row['Subject']
    group = row['Group']
    if group not in grouped_data:
        grouped_data[group] = []
    grouped_data[group].append(data_dict[subj])
       
#%% Plotting --  Create a single plot with three subplots for each group, within each subplot, plot all three gap conditions 

sns.set_palette ("Dark2")

variable_sets = ['gapcap16_sel', 'gapcap32_sel', 'gapcap64_sel']

groups = ['YNH', 'MNH', 'TTS']

t = data_dict[subjlist[0]]['t']

fig, ax = plt.subplots(len(groups), 1, sharex=True, sharey=True, figsize=(8,6))
fig.suptitle('GDT | Light Sedation - EEG Cap (Picks = A22, A23, A24)', fontsize=16)
fig.subplots_adjust(top=0.99, hspace=0.35)

gaps = ['16 ms', '32 ms', '64 ms']

for i, group in enumerate(groups):
    ax[i].set_title(f'{group}', fontsize=12)

    variable_data = []

    for subj in grouped_data[group]:
        # List to store the variable data for the current subject
        subj_variable_data = []

        for variable in variable_sets:
            subj_variable_data.append(subj[variable])

        variable_data.append(subj_variable_data)

    # Calculate the mean within each chin, and across chins for each variable in the current group
    variable_means = [np.mean(variable, axis=0)*1e6 for variable in variable_data]
    variable_sems = [stats.sem(variable)*1e6 for variable in variable_data]

    # Plot the variables on the current subplot
    for mean, sem, variable, gap in zip(variable_means, variable_sems, variable_sets, gaps):
        ax[i].plot(t, mean, label= gap)
        ax[i].fill_between(t, (mean - sem), (mean + sem), alpha=0.3)

    ax[i].set_xlim(-0.2,1.1)
    ax[i].set_ylim(-2,2)
    ax[i].grid()
    
    # # Labeling the stim on, off, gap trigger 
    # for x_value in (0, 2) :
    #     ax[i].axvline(x=x_value, color='black', linestyle='--', alpha=1)
           
    # ax[i].axvline(x=1, color='blue', linestyle='--', alpha=1)
    
# y_limits = ax[0].get_ylim()
# labels = ["Stim On", "Gap", "Stim Off"]
# for x, label in zip([0,1,2], labels):
#     ax[0].text(x, y_limits[1] + 0.05e-4, label, ha='center',weight='bold')
        
ax[0].legend(loc='upper right')

plt.xlabel('Time (s)', fontsize=12)
ax[1].set_ylabel('Amplitude (\u03bcV)', fontsize=12)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

plt.savefig(fig_loc + "GDT_LightSedationSelChans2_AcrossChins_Gap.png", dpi=500, bbox_inches="tight")

gap64_cap = []
gap64_mastoid = []
gap64_vertex = []
gap64_ground = []
evoked64_cap = []
evoked64_mastoid = []
evoked64_vertex = []
evoked64_ground = []

for group in groups:
    group_gap64cap = []  # Initialize lists for each condition and age group
    group_gap64mastoid = []
    group_gap64vertex = []
    group_gap64ground = []
    group_evoked64cap = []  
    group_evoked64mastoid = []
    group_evoked64vertex = []
    group_evoked64ground = []

    for column in dat:
        dat1 = io.loadmat(data_loc + sub + '_64ms_2-20Hz.mat', squeeze_me=True)
        dat1.keys()
        picks = dat1['picks']
        gap64cap = dat1['gap_cap']
        gap64mastoid = dat1['gap_mastoid']
        gap64vertex = dat1['gap_vertex']
        gap64ground = dat1['gap_ground']
        evoked64cap = dat1['ep_all']
        evoked64mastoid = dat1['ep_mastoid']
        evoked64vertex = dat1['ep_vertex']
        evoked64ground = dat1['ep_ground']

        group_gap64cap.append(gap64cap.mean(axis=0))
        group_gap64mastoid.append(gap64mastoid.mean(axis=0))
        group_gap64vertex.append(gap64vertex.mean(axis=0))
        group_gap64ground.append(gap64ground.mean(axis=0))
        group_evoked64cap.append(evoked64cap.mean(axis=0))  
        group_evoked64mastoid.append(evoked64mastoid.mean(axis=0))
        group_evoked64vertex.append(evoked64vertex.mean(axis=0))
        group_evoked64ground.append(evoked64ground.mean(axis=0))

    # Append data for each age group to lists
    gap64_cap.append(group_gap64cap)
    gap64_mastoid.append(group_gap64mastoid)
    gap64_vertex.append(group_gap64vertex)
    gap64_ground.append(group_gap64ground)
    evoked64_cap.append(group_evoked64cap)
    evoked64_mastoid.append(group_evoked64mastoid)
    evoked64_vertex.append(group_evoked64vertex)
    evoked64_ground.append(group_evoked64ground