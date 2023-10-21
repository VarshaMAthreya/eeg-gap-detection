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
    
    selected_rows = [0, 2, 4]  # Specify the rows you want to include
    gapcap16_mean = np.mean(mat_data['gap_cap16'][selected_rows], axis=0)


    data_dict[subj] = { 'picks' : mat_data['picks'],
                        'gapcap16' : mat_data['gap_cap16'],
                        'gapcap32' : mat_data['gap_cap32'],
                        'gapcap64' : mat_data['gap_cap64'],
                        'gapground16' : mat_data['gap_ground16'],
                        'gapground32' : mat_data['gap_ground32'],
                        'gapground64' : mat_data['gap_ground64'],
                        'gapmastoid16' : mat_data['gap_mastoid16'],
                        'gapmastoid32' : mat_data['gap_mastoid32'],
                        'gapmastoid64' : mat_data['gap_mastoid64'],
                        'gapvertex16' : mat_data['gap_vertex16'],
                        'gapvertex32' : mat_data['gap_vertex32'],
                        'gapvertex64' : mat_data['gap_vertex64'],
                        'cap16' : mat_data['cap16'],
                        'cap32' : mat_data['cap32'],
                        'cap64' : mat_data['cap64'],
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
                        't_full' : mat_data['t_full'] }
                              
##Organizing mat data by groups 
grouped_data = {}
for index, row in dat.iterrows():
    subj = row['Subject']
    group = row['Group']
    if group not in grouped_data:
        grouped_data[group] = []
    grouped_data[group].append(data_dict[subj])
       
#%% Plotting 

sns.set_palette ("Dark2")

variable_sets = ['cap16', 'cap32', 'cap64']

groups = ['YNH', 'MNH', 'TTS']

t = data_dict[subjlist[0]]['t_full']

# Create a single plot with three subplots for each group, within each subplot, plot all three gap conditions 
fig, ax = plt.subplots(len(groups), 1, sharex=True, sharey=True, figsize=(8,6))
fig.suptitle('GDT | Light Sedation - EEG Cap', fontsize=16)
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
    variable_means = [np.mean((np.mean(variable, axis=0)*1e6),axis=0) for variable in variable_data]
    variable_sems = [stats.sem((np.mean(variable, axis=0)*1e6), axis=0) for variable in variable_data]

    # Plot the variables on the current subplot
    for mean, sem, variable, gap in zip(variable_means, variable_sems, variable_sets, gaps):
        ax[i].plot(t, mean, label= gap)
        ax[i].fill_between(t, (mean - sem), (mean + sem), alpha=0.3)

    ax[i].set_xlim(-0.2,2.1)
    ax[i].set_ylim(-3.2,2.3)
    ax[i].grid()
    
    # Labeling the stim on, off, gap trigger 
    for x_value in (0, 2) :
        ax[i].axvline(x=x_value, color='black', linestyle='--', alpha=1)
           
    ax[i].axvline(x=1, color='blue', linestyle='--', alpha=1)
    
y_limits = ax[0].get_ylim()
labels = ["Stim On", "Gap", "Stim Off"]
for x, label in zip([0,1,2], labels):
    ax[0].text(x, y_limits[1] + 0.05e-4, label, ha='center',weight='bold')
        
ax[0].legend(loc='upper right')

plt.xlabel('Time (s)', fontsize=12)
ax[1].set_ylabel('Amplitude (\u03bcV)', fontsize=12)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

plt.savefig(fig_loc + "GDT_LightSedation_AcrossChins_Full2sec.png", dpi=500, bbox_inches="tight")


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
    evoked64_ground.append(group_evoked64ground)

#%%##Plotting 

sns.set_palette ("Dark2")

# fig, ax = plt.subplots(1, 1, sharex=True, sharey=True)
# ax.plot(t,  ynh_ep_mean16, label='16 ms', alpha=0.9)
# ax.fill_between(t,  ynh_ep_mean16 - ynh_ep_sem16,  ynh_ep_mean16 + ynh_ep_sem16, alpha=0.3)
# ax.plot(t,  ynh_ep_mean32, label='32 ms', alpha=0.9)
# ax.fill_between(t,  ynh_ep_mean32 - ynh_ep_sem32,  ynh_ep_mean32 + ynh_ep_sem32, alpha=0.3)
# ax.plot(t,  ynh_ep_mean64, label='64 ms', alpha=0.9)
# ax.fill_between(t,  ynh_ep_mean64 - ynh_ep_sem64,  ynh_ep_mean64 + ynh_ep_sem64, alpha=0.3)

# y_limits = ax.get_ylim()
# ax.axvline(x=0, color = 'black',linestyle='--', alpha=0.7)
# ax.text(0, y_limits[1] + 0.01, 'Stim On', ha='center', weight='bold')
# ax.axvline(x=1, color='blue', linestyle='--', alpha=0.7)
# ax.text(1, y_limits[1] + 0.01, 'Gap', ha='center')

# plt.xlim([-0.2, 2.1])
# # plt.ylim([0.5,2])
# ax.legend(prop={'size': 8})
# plt.xlabel('Time (in seconds)',fontsize=12)
# fig.text(0.0001, 0.5, 'Amplitude (\u03bcV)', va='center', rotation='vertical',fontsize=12)
# plt.suptitle('Cortical GDT | Light Sedation (N=3)', fontsize=14)
# plt.rcParams["figure.figsize"] = (6.5, 6)
# plt.tight_layout(rect=[0, 0.03, 1, 0.95])
# plt.show()

# plt.savefig(fig_loc + 'ChinGDT_AcrossGroups_1.png', dpi=400)

fig, ax = plt.subplots(3, 1, sharex=True, sharey=True)
ax[0].errorbar(t_full, ynh_ep_mean16*1e6, yerr=ynh_ep_sem16*1e6, linewidth=1,label='16 ms' +str(picks))
ax[0].errorbar(t_full, ynh_ep_mean32*1e6, yerr=ynh_ep_sem32*1e6, linewidth=1, label='32 ms')
ax[0].errorbar(t_full, ynh_ep_mean64*1e6, yerr=ynh_ep_sem64*1e6,linewidth=1, label='64 ms')
# ax[0].vlines(x=[1], ymin=(-6-0.5), ymax= (4+0.5), colors='black', ls='--')
ax[0].set_title('YNH - EEG Cap', loc='center', fontsize=12)

ax[1].errorbar(t_full, mnh_ep_mean16*1e6, yerr=mnh_ep_sem16*1e6,linewidth=1, label='16 ms' +str(picks))
ax[1].errorbar(t_full, mnh_ep_mean32*1e6, yerr=mnh_ep_sem32*1e6)
ax[1].errorbar(t_full, mnh_ep_mean64*1e6, yerr=mnh_ep_sem64*1e6)
# ax[1].vlines(x=[1], ymin=(-6-0.5), ymax= (4+0.5), colors='black', ls='--')
ax[1].set_title('MNH - EEG Cap', loc='center', fontsize=12)

ax[2].errorbar(t_full, tts_ep_mean16*1e6, yerr=tts_ep_sem16*1e6)
ax[2].errorbar(t_full, tts_ep_mean32*1e6, yerr=tts_ep_sem32*1e6)
ax[2].errorbar(t_full, tts_ep_mean64*1e6, yerr=tts_ep_sem64*1e6)
# ax[2].vlines(x=[1], ymin=(-6-0.5), ymax= (4+0.5), colors='black', ls='--')
ax[2].set_title('TTS - EEG Cap', loc='center', fontsize=12)

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
# plt.ylim([0.5,2])
ax[0].legend(prop={'size': 8})
plt.xlabel('Time (in seconds)',fontsize=12)
fig.text(0.0001, 0.5, 'Amplitude (\u03bcV)', va='center', rotation='vertical',fontsize=12)
plt.suptitle('Cortical GDT | Light Sedation', fontsize=14)
plt.rcParams["figure.figsize"] = (6.5, 6)
# plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

