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
fig_loc = 'C:/Users/vmysorea/Desktop/PhD/GreenLightMeeting/Figures/'
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

def group_age(age):
    if age <= 35:
        return 'YNH'
    elif age <= 55:
        return 'MNH'
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

# Define condition names
condition_names = { 0: 'Evoked - Gap 16 ms',
    1: 'Evoked - Gap 32 ms',
    2: 'Evoked - Gap 64 ms',
    3: 'ITC 16 ms',
    4: 'ITC 32 ms',
    5: 'ITC 64 ms',
    6: 'Onset - 16 ms',
    7: 'Onset - 32 ms',
    8: 'Onset - 64 ms'}

# Define age group labels
age_group_labels = {'YNH': 'Young (<=35 y)',
                    'MNH': 'Middle (36-55 y)',
                    'ONH': 'Old (>=56 y)'}

cond_groups = [(0,1,2), (3,4,5)]

# Color-blind friendly palette from seaborn
sns.set_palette("bright")

# Create a figure with 3 horizontal subplots
for cond in cond_groups:
    fig, axs = plt.subplots(3, 1, figsize=(4.5,4), sharex= True)
    
# Iterate through age groups
    for age_group_index, age_group in enumerate(age_group_labels.keys()):

        ax = axs[age_group_index]
        # N = age_groups['Subject'].count()
        ax.set_title(f'Age Group: {age_group_labels[age_group]}')

        # Iterate through conditions
        legend_text=[]
        for condition in cond:
            mean_age_group = mean_data[condition][age_group_index]
            sem_age_group = sem_data[condition][age_group_index]

            condition_name = condition_names.get(condition, f'Condition {condition}')
            
            N = age_groups['age_group'].count()[age_group]
            # Plot mean with SEM as shaded region
            ax.plot(t, mean_age_group, label=f'{condition_name}', alpha=0.7)
            ax.fill_between(t, mean_age_group - sem_age_group, mean_age_group + sem_age_group, alpha=0.3)
            
            legend_text.append(f"{age_group} (N={N})")

        if age_group_index == 0:
            ax.legend(loc ='upper right',fontsize = 'xx-small' )

        # ax.set_ylabel()
        # ax.set_ylim(-2,5.2)
        # ax.set_xlim(-0.1,1.1)
        ax.grid()

        fig.text(0, 0.5, 'Amplitude (\u03bcV)', va='center', rotation='vertical', fontsize=12)
        plt.xlabel('Time (s)', fontsize =12)
        # fig.suptitle(f'{condition_name}', size=16, y=1.001)

    plt.tight_layout()
    # plt.savefig(fig_loc + f'cond_{cond[0]}_{cond[1]}_1.png', dpi = 500)
    # plt.close()
    plt.show()  # Show the plot

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

# Color-blind friendly palette from seaborn
# sns.set_palette("Paired")

# age_colors = {
#     'YNH': '#1f78b4',
#     'MNH': '#6a3d9a',  # Second color in the palette
#     'ONH': '#33a02c'}  # Third color in the palette

# Define age group labels
age_group_labels = {'YNH': 'Young (<36 y)', 'MNH': 'Middle (36-55 y)', 'ONH': 'Old (>55 y)'}

# Create a figure with 3 subplots

fig, axs = plt.subplots(3, 1, figsize=(6.5, 5), sharex=True)

# Loop through conditions and plot in subplots
for condition_index, condition in enumerate([3, 4, 5]):
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
                
        ax.plot(t, mean_age_group, label=age_group, alpha=0.7, linewidth=2)
        ax.fill_between(t, mean_age_group - sem_age_group, mean_age_group + sem_age_group, alpha=0.2)

        legend_text.append(f"{age_group} (N={N})")

    if condition_index == 0:
        ax.legend(labels=legend_text, loc='upper right', fontsize='xx-small')

    ax.set_ylim(0.02,0.08)
    ax.set_xlim(-0.2,0.55)
    ax.grid()

plt.xlabel('Time (s)', fontsize=12)
fig.text(0, 0.5, 'Amplitude (\u03bcV)', va='center', rotation='vertical', fontsize=12)
fig.suptitle('Picks - A32', x=0.9, ha='right', fontsize=10)
fig.tight_layout()
# plt.subplots_adjust(wspace=0.1,hspace =0.1)
plt.show()

plt.savefig(fig_loc + 'ITC_AcrossAges_A32_1.png', dpi = 500, bbox_inches="tight")