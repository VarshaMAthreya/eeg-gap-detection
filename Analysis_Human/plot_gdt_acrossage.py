# -*- coding: utf-8 -*-
"""
Created on Sat Sep  2 19:22:05 2023

@author: vmysorea
"""
import seaborn as sns
import pandas as pd
from scipy.stats import sem
import numpy as np
from scipy import io
from scipy.io import savemat
from matplotlib import pyplot as plt
import warnings
import sys
sys.path.append('C:/Users/vmysorea/Documents/mne-python/')

plt.switch_backend('QT5Agg')  # Making the plots interactive (Scrollable)
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')
# Defining the dimensions and quality of figures
# plt.rcParams.update({'figure.autolayout': True})
plt.rcParams["figure.figsize"] = (5.5, 5)
plt.rcParams['figure.dpi'] = 500
plt.rc('xtick', labelsize=11)
plt.rc('ytick', labelsize=8)

# %%Setting up stuff
fig_loc = 'C:/Users/vmysorea/Desktop/PhD/Stim_Analysis/MTB_Analysis/GreenLight/'
data_loc = 'D:/PhD/Stim_Analysis/GapDetection_EEG/AnalyzedFiles_Figures/GDT_matfiles/'
csv_loc = 'C:/Users/vmysorea/Desktop/PhD/Stim_Analysis/MTB_Analysis/'

# subjlist = ['S104']
subjlist = ['S273', 'S069', 'S072', 'S078', 'S088',
            'S104', 'S105', 'S259', 'S260', 'S268',
            'S269', 'S270', 'S271', 'S274', 'S277',
            'S279', 'S280', 'S281', 'S282', 'S284',
            'S285', 'S288', 'S290', 'S303',
            'S305', 'S308', 'S310', 'S312', 'S337',
            'S339', 'S340', 'S341', 'S342', 'S344',
            'S345', 'S347', 'S352', 'S355', 'S358']


for subj in range(len(subjlist)):
    sub = subjlist[subj]
    dat1 = io.loadmat(data_loc + sub + '_evoked(4chan)_ITC_BInc.mat', squeeze_me=True)
    dat1.keys()
    t = dat1['t1']  # For GDT and ITC
    t_full = dat1['t']

# Load data with subjects' age and gender and pick subj only from subjlist
dat = pd.read_csv(csv_loc + 'subj_age_gender.csv')
dat0 = dat['Subject'].isin(subjlist)
dat = dat[dat0]

# Categorizing into different age groups

# def group_age(age):
#     if age <= 35:
#         return 'YNH'
#     elif age <= 55:
#         return 'MNH'
#     else:
#         return 'ONH'
    
def group_age(age):
    if age <= 40:
        return 'YNH'
    else:
        return 'ONH'

dat['age_group'] = dat['Age'].apply([group_age])
# Grouping into three age groups, not sorting alphabetically
age_groups = dat.groupby(['age_group'], sort=False)

# %% Loading all subjects' data

# Initialize empty lists to store data for different conditions and age groups
gap16_all = []
gap32_all = []
gap64_all = []
itc16_all = []
itc32_all = []
itc64_all = []
full16_all = []
full32_all = []
full64_all = []

mat_agegroups = []
picks = [3]  #If considering only A32

for age, groups in age_groups:
    group_gap16 = []  # Initialize lists for each condition and age group
    group_gap32 = []
    group_gap64 = []
    group_itc16 = []
    group_itc32 = []
    group_itc64 = []
    group_full16 = []
    group_full32 = []
    group_full64 = []

    for index, column in groups.iterrows():
        subj = column['Subject']
        dat = io.loadmat(data_loc + subj + '_evoked(4chan)_ITC_BInc.mat', squeeze_me=True)
        dat.keys()
        gap16 = dat['evoked_1'][picks]
        gap32 = dat['evoked_2'][picks]
        gap64 = dat['evoked_3'][picks]
        itc16 = dat['itc1'][picks]
        itc32 = dat['itc2'][picks]
        itc64 = dat['itc3'][picks]
        full16 = dat['onset_16'][picks]
        full32 = dat['onset_32'][picks]
        full64 = dat['onset_64'][picks]

        group_gap16.append((gap16.mean(axis=0))*1e6)
        group_gap32.append((gap32.mean(axis=0))*1e6)
        group_gap64.append((gap64.mean(axis=0))*1e6)
        group_itc16.append(itc16.mean(axis=0))
        group_itc32.append(itc32.mean(axis=0))
        group_itc64.append(itc64.mean(axis=0))
        group_full16.append((full16.mean(axis=0))*1e6)
        group_full32.append((full32.mean(axis=0))*1e6)
        group_full64.append((full64.mean(axis=0))*1e6)

    # Append data for each age group to lists
    gap16_all.append(group_gap16)
    gap32_all.append(group_gap32)
    gap64_all.append(group_gap64)
    itc16_all.append(group_itc16)
    itc32_all.append(group_itc32)
    itc64_all.append(group_itc64)
    full16_all.append(group_full16)
    full32_all.append(group_full32)
    full64_all.append(group_full64)

# %%
conditions = {0: gap16_all,
              1: gap32_all,
              2: gap64_all,
              3: itc16_all,
              4: itc32_all,
              5: itc64_all,
              6: full16_all,
              7: full32_all,
              8: full64_all}

mean_data = {}
sem_data = {}

for condition, evkds_all in conditions.items():
    mean_age_group = []
    sem_age_group = []

    for age_group_evkds in evkds_all:
        mean_subjects = (np.mean(age_group_evkds, axis=0))
        sem_subjects = (sem(age_group_evkds, axis=0))
        mean_age_group.append(mean_subjects)
        sem_age_group.append(sem_subjects)

    mean_data[condition] = mean_age_group
    sem_data[condition] = sem_age_group

# %% Subplots across conditions, with individual age plots

condition_names = { 0: 'Gap 16 ms',
                    1: 'Gap 32 ms',
                    2: 'Gap 64 ms',
                    3: 'ITC 16 ms',
                    4: 'ITC 32 ms',
                    5: 'ITC 64 ms',
                    6: 'Onset - 16 ms',
                    7: 'Onset - 32 ms',
                    8: 'Onset - 64 ms'}

# Define age group labels
# age_group_labels = {'YNH': 'Young (<=35 y)',
#                     'MNH': 'Middle (36-55 y)',
#                     'ONH': 'Old (>=56 y)'}

age_group_labels = {'YNH': 'Young (<=40 y)',
                    'ONH': 'Old (>41 y)'}

cond_groups = [(0,1,2)]

# Color-blind friendly palette from seaborn
sns.set_palette("Dark2")

# Create a figure with 3 horizontal subplots
for cond in cond_groups:
    fig, axs = plt.subplots(2, 1, figsize=(5.5,5), sharex= True)
    
# Iterate through age groups
    for age_group_index, age_group in enumerate(age_group_labels.keys()):

        ax = axs[age_group_index]
        N = age_groups['age_group'].count()[age_group]
        ax.set_title(f'{age_group_labels[age_group]} (N={N})', fontsize=11)

        # Iterate through conditions
        # legend_text=[]
        for condition in cond:
            mean_age_group = mean_data[condition][age_group_index]
            sem_age_group = sem_data[condition][age_group_index]

            condition_name = condition_names.get(condition, f'Condition {condition}')
            
            # Plot mean with SEM as shaded region
            ax.plot(t, mean_age_group, label=f'{condition_name}', alpha=0.7)
            ax.fill_between(t, mean_age_group - sem_age_group, mean_age_group + sem_age_group, alpha=0.3)
            
            # legend_text.append(f"{age_group} (N={N})")

        if age_group_index == 0:
            ax.legend(loc ='upper right',fontsize = 'xx-small' )

        # ax.set_ylabel()
        ax.set_ylim(-0.2,0.5)
        ax.set_xlim(-0.1,0.55)
        ax.grid()
        ax.axvline(x=0, color='blue', linestyle='--', alpha=0.7)
        
        fig.text(0, 0.5, 'Amplitude (\u03bcV)', va='center', rotation='vertical', fontsize=12)
        plt.xlabel('Time (s)', fontsize =12)
        # fig.suptitle(f'{condition_name}', size=16, y=1.001)
        
    fig.suptitle('Picks - Cz, Fz, FC1, FC2', x=1, ha='right', fontsize=10)
    plt.tight_layout()
    # plt.savefig(fig_loc + f'cond_{cond[0]}_{cond[1]}_1.png', dpi = 500)
    # plt.close()
    plt.show()  # Show the plot

plt.savefig(fig_loc + 'EvokedGapResponses_AcrossAge.png', dpi=500,bbox_inches="tight")
    

# %%## all three age groups in same subplot

condition_names = {0: 'Evoked - Gap 16 ms',
                   1: 'Evoked - Gap 32 ms',
                   2: 'Evoked - Gap 64 ms',
                   3: 'ITC 16 ms',
                   4: 'ITC 32 ms',
                   5: 'ITC 64 ms',
                   6: 'Onset - 16 ms',
                   7: 'Onset - 32 ms',
                   8: 'Onset - 64 ms'}


sns.set_palette("Dark2")

# Define age group labels
# age_group_labels = {'YNH': 'Young (<36 y)', 'MNH': 'Middle (36-55 y)', 'ONH': 'Old (>55 y)'}

age_group_labels = {'YNH': 'Young (<=40 y)',
                    'ONH': 'Old (>41 y)'}

# Create a figure with 3 subplots

fig, axs = plt.subplots(2, 1, figsize=(6.5, 5), sharex=True)

# Loop through conditions and plot in subplots
for condition_index, condition in enumerate([6, 7, 8]):
    ax = axs[condition_index]
    ax.set_title(condition_names[condition])

    # line_color = condition_colors.get(condition, 'k')
    legend_text = []

    # Iterate through age groups
    for age_group_index, age_group in enumerate(age_group_labels.keys()):
        mean_age_group = mean_data[condition][age_group_index]
        sem_age_group = sem_data[condition][age_group_index]

        # Plot mean with SEM as shaded region
        N = age_groups['age_group'].count()[age_group]
                
        ax.plot(t_full, mean_age_group, label=age_group, alpha=0.9, linewidth=2)
        ax.fill_between(t_full, mean_age_group - sem_age_group, mean_age_group + sem_age_group, alpha=0.2)

        legend_text.append(f"{age_group} (N={N})")

    if condition_index == 0:
        ax.legend(labels=legend_text, loc='upper right', fontsize='xx-small')

    # ax.set_ylim(-0.2,0.45)    
    # ax.set_ylim(0.02,0.08)
    ax.set_xlim(-0.2,0.55)
    ax.grid()

plt.xlabel('Time (s)', fontsize=12)
fig.text(0, 0.5, 'Amplitude (\u03bcV)', va='center', rotation='vertical', fontsize=12)
fig.suptitle('Picks - Cz, Fz, FC1, FC2', x=1, ha='right', fontsize=10)
fig.tight_layout()
# plt.subplots_adjust(wspace=0.1,hspace =0.1)
plt.show()

plt.savefig(fig_loc + 'GDTOnset_AcrossAges_4chan.png', dpi = 500, bbox_inches="tight")


#%% Plotting each condition separately -- Helpful for full time viewing 

condition_names = {0: 'Evoked - Gap 16 ms',
                   1: 'Evoked - Gap 32 ms',
                   2: 'Evoked - Gap 64 ms',
                   3: 'ITC 16 ms',
                   4: 'ITC 32 ms',
                   5: 'ITC 64 ms',
                   6: 'Onset - 16 ms',
                   7: 'Onset - 32 ms',
                   8: 'Onset - 64 ms'}

# Define age group labels
# age_group_labels = {'YNH': 'Young (<36)', 'MNH': 'Middle (36-55)', 'ONH': 'Old (>55)'}

sns.set_palette ("Dark2")

age_group_labels = {'YNH': 'Young (<=40 y)',
                    'ONH': 'Old (>41 y)'}

fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)

condition_to_plot = 8

# Plot the selected condition in the subplot
ax.set_title(condition_names[condition_to_plot])

legend_text = []

# Iterate through age groups
for age_group_index, age_group in enumerate(age_group_labels.keys()):
    mean_age_group = mean_data[condition_to_plot][age_group_index]
    sem_age_group = sem_data[condition_to_plot][age_group_index]

    N = age_groups['age_group'].count()[age_group]

    # Plot mean with SEM as shaded region
    ax.plot(t_full, mean_age_group, label=age_group, alpha=0.9)
    ax.fill_between(t_full, mean_age_group - sem_age_group, mean_age_group + sem_age_group, alpha=0.3)

    legend_text.append(f"{age_group} (N={N})")

ax.legend(labels=legend_text, loc='upper right', fontsize='xx-small')

# ax.set_ylim(-1, 5.1)
# ax.set_xlim(-0.2, 5.5)
# ax.grid()

plt.xlabel('Time (s)', fontsize=12)
plt.ylabel('Amplitude (\u03bcV)', fontsize=12)
# plt.suptitle('Picks - Cz, Fz, FC1, FC2', x=0.8, ha='right', fontsize=10)
plt.tight_layout()
plt.show()

# plt.savefig(fig_loc + "Binding20_FullTime_AcrossAge.png", dpi=500, bbox_inches="tight")

#%% PLotting one age group, one condition  -- Useful for full time, used now for showing P1, P2 peaks etc. 
condition_names = {0: 'Evoked - Gap 16 ms',
                   1: 'Evoked - Gap 32 ms',
                   2: 'Evoked - Gap 64 ms',
                   3: 'ITC 16 ms',
                   4: 'ITC 32 ms',
                   5: 'ITC 64 ms',
                   6: 'Onset - 16 ms',
                   7: 'Onset - 32 ms',
                   8: 'Onset - 64 ms'}

# Define age group labels
# age_group_labels = {'YNH': 'Young (<36)', 'MNH': 'Middle (36-55)', 'ONH': 'Old (>55)'}

sns.set_palette ("Dark2")

age_group_labels = {'YNH': 'Young (<=40 y)',
                    'ONH': 'Old (>41 y)'}

selected_age_group = 'YNH'

#Times for peak picking and plotting peaks
p1_start = -0.1
p1_end = 0.09
p2_start =0.091
p2_end = 0.3

fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)

condition_to_plot = 2

# Plot the selected condition in the subplot
# ax.set_title(condition_names[condition_to_plot])

legend_text = []

# Check if the selected age group exists in the age_group_labels dictionary
if selected_age_group in age_group_labels:
    age_group_index = list(age_group_labels.keys()).index(selected_age_group)
    mean_age_group = mean_data[condition_to_plot][age_group_index]
    sem_age_group = sem_data[condition_to_plot][age_group_index]

    N = age_groups['age_group'].count()[selected_age_group]

    # Plot mean with SEM as shaded region
    ax.plot(t, mean_age_group, label=selected_age_group, alpha=0.9)
    # ax.fill_between(t_full, mean_age_group - sem_age_group, mean_age_group + sem_age_group, alpha=0.3)
    
    # Find the indices within the P1 time window
    p1_indices = (t >= p1_start) & (t <= p1_end)
    
    # Find the maximum positive peak within the P1 time window
    p1_positive_peak_indices = np.where((p1_indices) & (mean_age_group > 0))
    if p1_positive_peak_indices[0].size > 0:
        p1_max_peak_index = p1_positive_peak_indices[0][np.argmax(mean_age_group[p1_positive_peak_indices])]
        p1_peak_time = t[p1_max_peak_index]
        p1_peak_amplitude = mean_age_group[p1_max_peak_index]
        ax.scatter(p1_peak_time, p1_peak_amplitude, color='red', marker='o', label='P1 Peak', s=50)
        ax.annotate('P1', (p1_peak_time, p1_peak_amplitude), textcoords="offset points", xytext=(0, 10), ha='center')
    else:
        print(f"No positive P1 peak found for {selected_age_group}.")
    
    # Find the indices within the P2 time window
    p2_indices = (t >= p2_start) & (t <= p2_end)
    
    # Find the maximum positive peak within the P2 time window
    p2_positive_peak_indices = np.where((p2_indices) & (mean_age_group > 0))
    if p2_positive_peak_indices[0].size > 0:
        p2_max_peak_index = p2_positive_peak_indices[0][np.argmax(mean_age_group[p2_positive_peak_indices])]
        p2_peak_time = t[p2_max_peak_index]
        p2_peak_amplitude = mean_age_group[p2_max_peak_index]
        ax.scatter(p2_peak_time, p2_peak_amplitude, color='black', marker='o', label='P2 Peak', s=50)
        ax.annotate('P2', (p2_peak_time, p2_peak_amplitude), textcoords="offset points", xytext=(0, 10), ha='center')
    else:
        print(f"No positive P2 peak found for {selected_age_group}.")
    # legend_text.append(f"{selected_age_group} (N={N})")
    
    #Draw a line with an arrow for the P1 time window
    ax.annotate('', xy=(p1_start, 0), xytext=(p1_end, 0),
                arrowprops=dict(arrowstyle='<->', lw=2, color='red', shrinkA=0, shrinkB=0))
    
    # Draw a line with an arrow for the P2 time window
    ax.annotate('', xy=(p2_start, 0), xytext=(p2_end, 0),
                arrowprops=dict(arrowstyle='<->', lw=2, color='black', shrinkA=0, shrinkB=0))
    
    y_limits = ax.get_ylim()
    
    ax.axvline(x=0, color='blue', linestyle='--', alpha=0.7)
    ax.text(0, y_limits[1] + 0.01, 'Trigger', ha='center')
    
else:
    print(f"Selected age group '{selected_age_group}' not found.")
    
   

# ax.legend(labels=legend_text, loc='upper right', fontsize='xx-small')

ax.set_xlim(-0.1, 0.55)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

plt.xlabel('Time (s)', fontsize=12)
plt.ylabel('Amplitude (\u03bcV)', fontsize=12)
plt.suptitle('Peak-picking Method', x=0.95,ha='right', fontsize=14)
plt.tight_layout()
plt.show()

plt.savefig(fig_loc + 'MarkingP1P2GDT.png', dpi=400)