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
# plt.rcParams["figure.figsize"] = (5.5,5)
plt.rcParams['figure.dpi'] = 120

# %%Setting up stuff
fig_loc = 'C:/Users/vmysorea/Desktop/PhD/FinalThesis/ThesisDoc/Figures/'
data_loc = 'D:/PhD/Data/Chin_Data/AnalyzedGDT_matfiles/'
csv_loc = 'C:/Users/vmysorea/Desktop/PhD/Stim_Analysis/MTB_Analysis/'

subjlist =[ 'Q363', 'Q364', 'Q368',
            'Q402',  'Q406', 'Q407', 
             'Q424', 'Q426'] 

# subjlist = ['Q351', 'Q402','Q412']
#             # 'Q363', 'Q364', 'Q365', 'Q368',
#             #  'Q404', 'Q406', 'Q407', 'Q410',
#             #  'Q422', 'Q424', 'Q426', 'Q428'] 

# Load data with subjects' age and gender and pick subj only from subjlist
dat = pd.read_csv(csv_loc + 'chin_group_age.csv')
dat0 = dat['Subject'].isin(subjlist)
dat = dat[dat0]
        
#Loading mat files for each subject and loading all the variables 
data_dict = {}
for index, row in dat.iterrows():
    subj = row['Subject']
    mat_data = io.loadmat(data_loc + subj + '_WithNoGap_2-20Hz.mat', squeeze_me=True)
    
    gapcap_sel = mat_data['gap_cap'][0:6]
    gapcap16_sel = mat_data['gap_cap16'][0:6]
    gapcap32_sel = mat_data['gap_cap32'][0:6]
    gapcap64_sel = mat_data['gap_cap64'][0:6]
    
    cap_sel = mat_data['cap'][0:6]
    cap16_sel = mat_data['cap16'][0:6]
    cap32_sel = mat_data['cap32'][0:6]
    cap64_sel = mat_data['cap64'][0:6]

    data_dict[subj] = { 'picks' : mat_data['picks'],
                        'gapcap' : (mat_data['gap_cap']).mean(axis=0),
                        'gapcap16' : (mat_data['gap_cap16']).mean(axis=0),
                        'gapcap32' : (mat_data['gap_cap32']).mean(axis=0),
                        'gapcap64' : (mat_data['gap_cap64']).mean(axis=0),
                        'gapground16' : mat_data['gap_ground16'],
                        'gapground32' : mat_data['gap_ground32'],
                        'gapground64' : mat_data['gap_ground64'],
                        'gapmastoid' : mat_data['gap_mastoid'],
                        'gapmastoid16' : mat_data['gap_mastoid16'],
                        'gapmastoid32' : mat_data['gap_mastoid32'],
                        'gapmastoid64' : mat_data['gap_mastoid64'],
                        'gapvertex' : mat_data['gap_vertex'],
                        'gapvertex16' : mat_data['gap_vertex16'],
                        'gapvertex32' : mat_data['gap_vertex32'],
                        'gapvertex64' : mat_data['gap_vertex64'],
                        'cap' : (mat_data['cap']).mean(axis=0),
                        'cap16' : (mat_data['cap16']).mean(axis=0),
                        'cap32' : (mat_data['cap32']).mean(axis=0),
                        'cap64' : (mat_data['cap64']).mean(axis=0),
                        'ground16' : mat_data['ground16'],
                        'ground32' : mat_data['ground32'],
                        'ground64' : mat_data['ground64'],
                        'mastoid' : mat_data['mastoid'],
                        'mastoid16' : mat_data['mastoid16'],
                        'mastoid32' : mat_data['mastoid32'],
                        'mastoid64' : mat_data['mastoid64'],
                        'vertex' : mat_data['vertex'],
                        'vertex16' : mat_data['vertex16'],
                        'vertex32' : mat_data['vertex32'],
                        'vertex64' : mat_data['vertex64'],
                        't' : mat_data['t'],
                        't_full' : mat_data['t_full'],
                        'gapcap_sel' : (gapcap_sel).mean(axis=0),
                        'gapcap16_sel' : (gapcap16_sel).mean(axis=0),
                        'gapcap32_sel' : (gapcap32_sel).mean(axis=0),
                        'gapcap64_sel' : (gapcap64_sel).mean(axis=0),
                        'cap_sel' : (cap_sel).mean(axis=0),
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
       
groups = ['YNH', 'MNH', 'TTS']

#%% Plotting --  Create a single plot with three subplots for each group, within each subplot, plot all three gap conditions 

sns.set_palette ("Dark2")

variable_sets = ['gapcap16_sel', 'gapcap32_sel', 'gapcap64_sel']

groups = ['YNH', 'MNH', 'TTS']

t = data_dict[subjlist[0]]['t']

fig, ax = plt.subplots(len(groups), 1, sharex=True, sharey=True, figsize=(12,8))
# fig.suptitle('(Picks = A7, A8, A9, A22, A23, A24)', fontsize=12)
fig.subplots_adjust(top=0.99, hspace=0.35)

gaps = ['16 ms', '32 ms', '64 ms']

legend_text = []

for i, group in enumerate(groups):
    N = len(grouped_data[group])

    ax[i].set_title(f'{group} (N={N})', fontsize=22, weight="bold")

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
    ax[i].tick_params(axis='both', which='major', labelsize=18)
    ax[i].grid()
    
    # # Labeling the stim on, off, gap trigger 
    
    # for x_value in (0,2) :
    #     ax[i].axvline(x=x_value, color='black', linestyle='--', alpha=1)
           
    ax[i].axvline(x=1, color='black', linestyle='--', alpha=1, linewidth=2)
    ax[i].axvline(x=0, color='blue', linestyle='--', alpha=1, linewidth=2)
    
    ax[i].fill_between(x=[0,0.55], y1=-2.5, y2=2.5, color='gray', alpha=0.2)
    
y_limits = ax[0].get_ylim()
# labels = ["Stim On", "Gap", "Stim Off"]
labels = ["End of Gap", "Stim Off"]
for x, label in zip([0,1], labels):
    ax[0].text(x, y_limits[1] + 0.05, label, ha='center', fontsize = 16)
        
ax[2].legend(loc='upper right', fontsize = 'x-large')

plt.xlabel('Time (s)', fontsize=22, weight='bold')
ax[1].set_ylabel('Amplitude (\u03bcV)', fontsize=22, weight='bold')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

plt.savefig(fig_loc + "GDT_AcrossGroups_1.png", dpi=500, bbox_inches="tight", transparent=True)

#%%Plotting full time 

sns.set_palette ("Dark2")

variable_sets = ['cap16_sel', 'cap32_sel', 'cap64_sel', 'cap_sel']

groups = ['YNH', 'MNH', 'TTS']

t = data_dict[subjlist[0]]['t_full']

fig, ax = plt.subplots(len(groups), 1, sharex=True, sharey=True, figsize=(8,6))
fig.suptitle('(Picks = A7, A8, A9, A22, A23, A24)', fontsize=12)
fig.subplots_adjust(top=0.99, hspace=0.35)

gaps = ['16 ms', '32 ms', '64 ms', 'No Gap']

legend_text = []

for i, group in enumerate(groups):
    N = len(grouped_data[group])

    ax[i].set_title(f'{group} (N={N})', fontsize=12, weight="bold")

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
        

    ax[i].set_xlim(-0.2,2.1)
    ax[i].set_ylim(-4,3)
    ax[i].grid()
    
    # # Labeling the stim on, off, gap trigger 
    
    for x_value in (0,2) :
        ax[i].axvline(x=x_value, color='black', linestyle='--', alpha=1)
           
    ax[i].axvline(x=1, color='blue', linestyle='--', alpha=1)
    
y_limits = ax[0].get_ylim()
labels = ["Stim On", "Gap", "Stim Off"]
# labels = ["Gap Trigger", "Stim Off"]
for x, label in zip([0,1,2], labels):
    ax[0].text(x, y_limits[1] + 0.05, label, ha='center',weight='bold')
        
ax[0].legend(loc='upper right', fontsize = 'x-small')

plt.xlabel('Time (s)', fontsize=12)
ax[1].set_ylabel('Amplitude (\u03bcV)', fontsize=12)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# plt.savefig(fig_loc + "GDT_LightSedationAllChans_AcrossChins_Full_SelChan.png", dpi=500, bbox_inches="tight")

#%% Plotting --  Create a single plot of gaps without comparison across groups

# # Combine data across groups
# combined_data = []

# for group in groups:
#     N = len(grouped_data[group])

#     for subj in grouped_data[group]:
#         combined_data.append(subj)

# # Plotting
# sns.set_palette("Dark2")

# variable_sets = ['gapcap16_sel', 'gapcap32_sel', 'gapcap64_sel']

# fig, ax = plt.subplots(1, 1, figsize=(8, 6))
# plt.title('(Picks = A7, A8, A9, A22, A23, A24)', fontsize=12)
# fig.subplots_adjust(top=0.95, hspace=0.35)

# gaps = ['16 ms', '32 ms', '64 ms']

# legend_text = []

# variable_data = []

# for subj in combined_data:
#     subj_variable_data = []

#     for variable in variable_sets:
#         subj_variable_data.append(subj[variable])

#     variable_data.append(subj_variable_data)

# variable_means = [np.mean(variable, axis=0) * 1e6 for variable in variable_data]
# variable_sems = [stats.sem(variable) * 1e6 for variable in variable_data]

# # Plot the variables on the current subplot
# for mean, sem, variable, gap in zip(variable_means, variable_sems, variable_sets, gaps):
#     ax.plot(t, mean, label=gap)
#     ax.fill_between(t, (mean - sem), (mean + sem), alpha=0.3)

# ax.set_xlim(-0.2, 1.1)
# ax.set_ylim(-3, 2)
# ax.grid()

# # Labeling the stim on, off, gap trigger
# ax.axvline(x=1, color='black', linestyle='--', alpha=1)
# ax.axvline(x=0, color='blue', linestyle='--', alpha=1)

# y_limits = ax.get_ylim()
# labels = ["End of gap", "Stim Off"]
# for x, label in zip([0, 1], labels):
#     ax.text(x, y_limits[1] + 0.05, label, ha='center', weight='bold')

# ax.legend(loc='upper right', fontsize='x-small')

# plt.xlabel('Time (s)', fontsize=12)
# plt.ylabel('Amplitude (\u03bcV)', fontsize=12)

# plt.tight_layout(rect=[0, 0.03, 1, 0.95])
# plt.show()

# plt.savefig(fig_loc + "Gap_LightSedation_AcrossGaps.png", dpi=500, bbox_inches="tight", transparent=True)

#%% Plotting --  Create a single plot with gap durations without comparison across groups -- Full time
t_full = mat_data['t_full']

# Combine data across groups
combined_data = []

for group in groups:
    N = len(grouped_data[group])

    for subj in grouped_data[group]:
        combined_data.append(subj)

# Subtracting subdermal electrodes 
subdermal64 = {}

for idx, data in enumerate(combined_data):
    vertex64 = data['vertex64']
    mastoid64 = data['mastoid64']
    
    data['subdermal64'] = vertex64 - mastoid64
    
# Subtracting subdermal electrodes 
subdermal32 = {}

for idx, data in enumerate(combined_data):
    vertex32 = data['vertex32']
    mastoid32 = data['mastoid32']
    
    data['subdermal32'] = vertex32 - mastoid32
    
# Subtracting subdermal electrodes 
subdermal16 = {}

for idx, data in enumerate(combined_data):
    vertex16 = data['vertex16']
    mastoid16 = data['mastoid16']
    
    data['subdermal16'] = vertex16 - mastoid16
        
# # Plotting

# # custom_palette = ["green", "purple"]  
# # sns.set_palette(custom_palette)

sns.set_palette("Dark2")

variable_sets = ['cap16_sel', 'cap32_sel', 'cap64_sel', 'cap_sel']

fig, ax = plt.subplots(1, 1, figsize=(9, 7))
ax.set_title('(Picks = A7, A8, A9, A22, A23, A24)', fontsize=12, pad=20)
# ax.set_title('16 ms', fontsize=26, pad=20)
fig.subplots_adjust(top=0.95, hspace=0.35)

gaps = ['16 ms', '32 ms', '64 ms', 'No Gap']

legend_text = []

variable_data = []

for subj in combined_data:
    subj_variable_data = []

    for variable in variable_sets:
        subj_variable_data.append(subj[variable])

    variable_data.append(subj_variable_data)

variable_means = [np.mean(variable, axis=0) for variable in variable_data]
variable_sems = [stats.sem(variable) for variable in variable_data]

# Plot the variables on the current subplot
for mean, sem, variable, gap in zip(variable_means, variable_sems, variable_sets, gaps):
    ax.plot(t_full, mean*1e6, label=gap, linewidth=2)
    ax.fill_between(t_full, (mean*1e6 - sem*1e6), (mean*1e6 + sem*1e6), alpha=0.2)

ax.set_xlim(-0.2, 2.2)
ax.set_ylim(-3.5, 3)
ax.grid()

# Labeling the stim on, off, gap trigger
ax.axvline(x=0, color='black', linestyle='--', alpha=2)
ax.axvline(x=2, color='black', linestyle='--', alpha=2)
ax.axvline(x=1, color='blue', linestyle='--', alpha=2)

y_limits = ax.get_ylim()
labels = ["Stim On", "End of gap", "Stim Off"]
for x, label in zip([0, 1, 2], labels):
    ax.text(x, y_limits[1] + 0.08, label, ha='center', weight='bold', fontsize=16)

ax.legend(loc='upper right', fontsize='large')

# ax.fill_between(x=[1.15,1.3], y1=-3.5, y2=3.5, color='steelblue', alpha=0.3)
ax.fill_between(x=[1,1.55], y1=-3.5, y2=3, color='gray', alpha=0.2)

plt.xlabel('Time (s)', fontsize=22)
plt.ylabel('Amplitude (\u03bcV)', fontsize=22)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

plt.savefig(fig_loc + "GDT_WithNoGaps_ALlChins_1.png", dpi=500, bbox_inches="tight", transparent=True)


#%% Plotting between subdermal and cap electrodes
# t_full = mat_data['t_full']

# # Combine data across groups
# combined_data_sb = []

# for group in groups:
#     N = len(grouped_data[group])

#     for subj in grouped_data[group]:
#         combined_data_sb.append(subj)

# # Subtracting subdermal electrodes 
# subdermal64 = {}

# for idx, data in enumerate(combined_data_sb):
#     vertex64 = data['vertex64']
#     mastoid64 = data['mastoid64']
    
#     data['subdermal64'] = -vertex64 + mastoid64
    
# # Subtracting subdermal electrodes 
# subdermal32 = {}

# for idx, data in enumerate(combined_data_sb):
#     vertex32 = data['vertex32']
#     mastoid32 = data['mastoid32']
    
#     data['subdermal32'] = -vertex32 + mastoid32
    
# # Subtracting subdermal electrodes 
# subdermal16 = {}

# for idx, data in enumerate(combined_data_sb):
#     vertex16 = data['vertex16']
#     mastoid16 = data['mastoid16']
    
#     data['subdermal16'] = -vertex16 + mastoid16
        
# # Set the seaborn style and palette
# sns.set_style("whitegrid")
# sns.set_palette(["green", "purple"])

# # Define the variables and groups
# variable_sets = ['cap16_sel', 'cap32_sel', 'cap64_sel']
# subdermal_sets = ['subdermal16', 'subdermal32', 'subdermal64']
# t_full = mat_data['t_full']

# titles = ['16 ms', '32 ms', '64 ms']

# # Create subplots
# fig, axes = plt.subplots(3, 1, sharey=True, figsize=(12, 8))
# fig.subplots_adjust(top=0.99, hspace=0.35)

# legend_text = []

# for i, (variable, subdermal, title) in enumerate(zip(variable_sets, subdermal_sets, titles)):
#     axes[i].set_title(title, fontsize=16, weight="bold", pad=16)  # Set custom title for each subplot

#     variable_data = np.array([data[variable] for data in combined_data_sb])
#     subdermal_data = np.array([data[subdermal] for data in combined_data_sb])

#     # Calculate the mean and SEM across subjects for each variable
#     variable_means = np.mean(variable_data, axis=0)
#     variable_sems = stats.sem(variable_data, axis=0)

#     subdermal_means = np.mean(subdermal_data, axis=0) 
#     subdermal_sems = stats.sem(subdermal_data, axis=0) 

#     # Plot the variables on the current subplot
#     axes[i].plot(t_full, variable_means  *1e6 , label='EEG Cap (N=15)', linewidth=2)
#     axes[i].fill_between(t_full, (variable_means *1e6- variable_sems *1e6), (variable_means *1e6 + variable_sems *1e6), alpha=0.2)
    
#     axes[i].plot(t_full, subdermal_means *1e6, label='Subdermal Electrodes (N=15)', linewidth=2)
#     axes[i].fill_between(t_full, (subdermal_means *1e6 - subdermal_sems *1e6), (subdermal_means *1e6 + subdermal_sems *1e6), alpha=0.2)
    
#     axes[i].grid()
    
#     if i == 0:
#         axes[i].legend(loc='upper right', fontsize='large')
    
#     axes[i].set_xlim(-0.2,2.1)
#     axes[i].set_ylim(-3,3)
#     axes[i].tick_params(axis='both', which='major', labelsize=16)
    
#     # # Labeling the stim on, off, gap trigger
#     axes[i].axvline(x=0, color='black', linestyle='--', alpha=2)
#     axes[i].axvline(x=2, color='black', linestyle='--', alpha=2)
#     axes[i].axvline(x=1, color='blue', linestyle='--', alpha=2)

#     y_limits = axes[i].get_ylim()
#     labels = ["Stim On", "End of gap", "Stim Off"]
#     for x, label in zip([0, 1, 2], labels):
#         axes[0].text(x, y_limits[1] + 0.06, label, ha='center', weight='bold', fontsize=14)
        
#     axes[i].fill_between(x=[1,1.55], y1=-3, y2=3, color='gray', alpha=0.2)
    
# plt.xlabel('Time (s)', fontsize=20, weight ='bold')
# axes[1].set_ylabel('Amplitude (\u03bcV)', fontsize=22, weight ='bold')


# plt.tight_layout(rect=[0, 0.03, 1, 0.95])
# plt.show()

# plt.savefig(fig_loc + "GDT_SBvsCap_AllGaps_3.png", dpi=500, bbox_inches="tight", transparent=True)
#%% Plotting - Does not work 

variable_sets = [('cap16_sel', 'subdermal16'), ('cap32_sel', 'subdermal32'), ('cap64_sel', 'subdermal64')]
gaps = ['16 ms', '32 ms', '64 ms']

fig, axes = plt.subplots(3, 1, sharex = True, figsize=(12, 8))  # 3 rows, 1 column for each subplot
fig.subplots_adjust(top=0.95, hspace=0.35)

legend_names = ['EEG Cap (N=15)', 'Subdermal (N=15)']

for ax, (cap_variable, subdermal_variable), gap in zip(axes, variable_sets, gaps):
    ax.set_title(gap, fontsize=26, pad=16)

    variable_data = []
    for subj in combined_data:
        subj_variable_data = [subj[cap_variable], subj[subdermal_variable]]
        variable_data.append(subj_variable_data)

    variable_means = [np.mean(variable, axis=0) for variable in variable_data]
    variable_sems = [stats.sem(variable) for variable in variable_data]

    for mean, sem, variable in zip(variable_means, variable_sems, [cap_variable, subdermal_variable]):
        ax.plot(t_full, mean *1e6, label=variable, linewidth=4)
        ax.fill_between(t_full, (mean*1e6 - sem*1e6), (mean*1e6 + sem*1e6), alpha=0.3)

    ax.set_xlim(-0.2, 2.2)
    ax.set_ylim(-3, 3)
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.grid()

    ax.axvline(x=0, color='black', linestyle='--', alpha=2)
    ax.axvline(x=2, color='black', linestyle='--', alpha=2)
    ax.axvline(x=1, color='blue', linestyle='--', alpha=2)

    y_limits = ax.get_ylim()
    labels = ["Stim On", "End of gap", "Stim Off"]
    for x, label in zip([0, 1, 2], labels):
        axes[0].text(x, y_limits[1] + 0.08, label, ha='center', weight='bold', fontsize=18)
    
    ax.fill_between(x=[1,1.55], y1=-3, y2=3, color='gray', alpha=0.2)

    plt.xlabel('Time (s)', fontsize=22, weight='bold')
    axes[1].set_ylabel('Amplitude (\u03bcV)', fontsize=22, weight='bold')
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)

# Create the legend and place it only on the top-most subplot
axes[0].legend(legend_names, loc='upper right', fontsize='large')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

plt.savefig(fig_loc + "GDT_SBvsCap_AllGaps_Trial5.png", dpi=500, bbox_inches="tight", transparent=True)
