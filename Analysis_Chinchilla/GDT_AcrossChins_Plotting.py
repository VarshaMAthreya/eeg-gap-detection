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
import pandas as pd


plt.switch_backend('QT5Agg')  # Making the plots interactive (Scrollable)
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')
# Defining the dimensions and quality of figures
plt.rcParams["figure.figsize"] = (5.5,5)
plt.rcParams['figure.dpi'] = 120

# %%Setting up stuff
fig_loc = 'C:/Users/vmysorea/Desktop/PhD/Stim_Analysis/MTB_Analysis/GreenLight/'
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

group_data = {'YNH': [], 'MNH': [], 'TTS': []}

for group, data_list in grouped_data.items():
    for data in data_list: 
        gapcap16_mean = data['gapcap16'].mean(axis=0)
        gapcap32_mean = data['gapcap32'].mean(axis=0)
        gapcap64_mean = data['gapcap64'].mean(axis=0)
        cap16_mean = data['cap16'].mean(axis=0)
        cap32_mean = data['cap32'].mean(axis=0)
        cap64_mean = data['cap64'].mean(axis=0)  
        
        group_data[group].append(gapcap16_mean)
        group_data[group].append(gapcap32_mean)
        group_data[group].append(gapcap64_mean)
        group_data[group].append(cap16_mean)
        group_data[group].append(cap32_mean)
        group_data[group].append(cap64_mean)
    
#%% Getting data from individual mat files across gap conditions 
# Initialize empty lists to store data for different conditions and age groups
gapcap16 = []
gapmastoid16 = []
gapvertex16 = []
gapground16 = []
evokedcap16 = []
evokedmastoid16 = []
evokedvertex16 = []
evokedground16 = []

# for group in groups:
group_gap16cap = []  # Initialize lists for each condition and age group
group_gap16mastoid = []
group_gap16vertex = []
group_gap16ground = []
group_evoked16cap = []  
group_evoked16mastoid = []
group_evoked16vertex = []
group_evoked16ground = []

for index, column in dat.iterrows():
    subj = column['Subject']
    dat1 = io.loadmat(data_loc + subj + '_AllGaps_2-20Hz.mat', squeeze_me=True)
    dat1.keys()
    picks = dat1['picks']
    gap16cap = dat1['gap_cap16']
    gap16mastoid = dat1['gap_mastoid16']
    gap16vertex = dat1['gap_vertex16']
    gap16ground = dat1['gap_ground16']
    # evoked16cap = dat1['ep_all']
    # evoked16mastoid = dat1['ep_mastoid']
    # evoked16vertex = dat1['ep_vertex']
    # evoked16ground = dat1['ep_ground']

    group_gap16cap.append(gap16cap.mean(axis=0))
    group_gap16mastoid.append(gap16mastoid.mean(axis=0))
    group_gap16vertex.append(gap16vertex.mean(axis=0))
    group_gap16ground.append(gap16ground.mean(axis=0))
    # group_evoked16cap.append(evoked16cap.mean(axis=0))  
    # group_evoked16mastoid.append(evoked16mastoid.mean(axis=0))
    # group_evoked16vertex.append(evoked16vertex.mean(axis=0))
    # group_evoked16ground.append(evoked16ground.mean(axis=0))

# Append data for each age group to lists
# gap16cap.append(group_gap16cap)
# gap16_mastoid.append(group_gap16mastoid)
# gap16_vertex.append(group_gap16vertex)
# gap16_ground.append(group_gap16ground)
# evoked16_cap.append(group_evoked16cap)
# evoked16_mastoid.append(group_evoked16mastoid)
# evoked16_vertex.append(group_evoked16vertex)
# evoked16_ground.append(group_evoked16ground)

#%% 32ms Gap 

# Initialize empty lists to store data for different conditions and age groups
gap32_cap = []
gap32_mastoid = []
gap32_vertex = []
gap32_ground = []
evoked32_cap = []
evoked32_mastoid = []
evoked32_vertex = []
evoked32_ground = []

for group in groups:
    group_gap32cap = []  # Initialize lists for each condition and age group
    group_gap32mastoid = []
    group_gap32vertex = []
    group_gap32ground = []
    group_evoked32cap = []  
    group_evoked32mastoid = []
    group_evoked32vertex = []
    group_evoked32ground = []

    for column in dat:
        dat1 = io.loadmat(data_loc + sub + '_32ms_2-20Hz.mat', squeeze_me=True)
        dat1.keys()
        picks = dat1['picks']
        gap32cap = dat1['gap_cap']
        gap32mastoid = dat1['gap_mastoid']
        gap32vertex = dat1['gap_vertex']
        gap32ground = dat1['gap_ground']
        evoked32cap = dat1['ep_all']
        evoked32mastoid = dat1['ep_mastoid']
        evoked32vertex = dat1['ep_vertex']
        evoked32ground = dat1['ep_ground']

        group_gap32cap.append(gap32cap.mean(axis=0))
        group_gap32mastoid.append(gap32mastoid.mean(axis=0))
        group_gap32vertex.append(gap32vertex.mean(axis=0))
        group_gap32ground.append(gap32ground.mean(axis=0))
        group_evoked32cap.append(evoked32cap.mean(axis=0))  
        group_evoked32mastoid.append(evoked32mastoid.mean(axis=0))
        group_evoked32vertex.append(evoked32vertex.mean(axis=0))
        group_evoked32ground.append(evoked32ground.mean(axis=0))

    # Append data for each age group to lists
    gap32_cap.append(group_gap32cap)
    gap32_mastoid.append(group_gap32mastoid)
    gap32_vertex.append(group_gap32vertex)
    gap32_ground.append(group_gap32ground)
    evoked32_cap.append(group_evoked32cap)
    evoked32_mastoid.append(group_evoked32mastoid)
    evoked32_vertex.append(group_evoked32vertex)
    evoked32_ground.append(group_evoked32ground)

#%% 64 ms 

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


#%%Loading mat files -- Need to work on this later

subjlist_ynh = ['Q412', 'Q422', 'Q424', 'Q426', 'Q428'] 

for subj in range(len(subjlist_ynh)):
    sub = subjlist_ynh[subj]
    dat1 = io.loadmat(data_loc + sub + '_16ms_2-20Hz.mat', squeeze_me=True)
    dat2 = io.loadmat(data_loc + sub + '_32ms_2-20Hz.mat', squeeze_me=True)
    dat3 = io.loadmat(data_loc + sub + '_64ms_2-20Hz.mat', squeeze_me=True)

    dat1.keys()
    dat2.keys()
    dat3.keys()
    
    ynh_ep_mastoid16 = dat1['ep_mastoid']
    ynh_ep_vertex16 = dat1['ep_vertex']
    ynh_ep_ground16= dat1['ep_ground']
    ynh_ep_all16 = dat1['ep_all']
    # ynh_ep_mean16 = (dat1['ep_mean'])*1e6
    # ynh_ep_sem16 = (dat1['ep_sem'])*1e6
    # ynh_ep_subderm16 = dat1['ep_subderm']
    # ynh_ep_mean_subderm16 = dat1['ep_mean_subderm']
    # ynh_ep_sem_subderm16 = dat1['ep_sem_subderm']
    picks= dat1['picks']
    t=dat1['t']
    t_full=dat1['t_full']
    
    ynh_ep_mastoid32 = dat2['ep_mastoid']
    ynh_ep_vertex32 = dat2['ep_vertex']
    ynh_ep_ground32= dat2['ep_ground']
    ynh_ep_all32 = dat2['ep_all']
    # ynh_ep_mean32 = (dat2['ep_mean'])*1e6
    # ynh_ep_sem32 = (dat2['ep_sem'])*1e6
    # ynh_ep_subderm32 = dat2['ep_subderm']
    # ynh_ep_mean_subderm32 = dat2['ep_mean_subderm']
    # ynh_ep_sem_subderm32 = dat2['ep_sem_subderm']
    
    ynh_ep_mastoid64 = dat3['ep_mastoid']
    ynh_ep_vertex64 = dat3['ep_vertex']
    ynh_ep_ground64= dat3['ep_ground']
    ynh_ep_all64 = dat3['ep_all']
    # ynh_ep_mean64 = (dat3['ep_mean'])*1e6
    # ynh_ep_sem64 = (dat3['ep_sem'])*1e6
    # ynh_ep_subderm64 = dat3['ep_subderm']
    # ynh_ep_mean_subderm64 = dat3['ep_mean_subderm']
    # ynh_ep_sem_subderm64 = dat3['ep_sem_subderm']
    
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

ynh_ep_mean16 = ynh_ep_all16.mean(axis=0)
ynh_ep_sem16 = ynh_ep_all16.std(axis=0) / np.sqrt(ynh_ep_all16.shape[0])

ynh_ep_mean32 = ynh_ep_all32.mean(axis=0)
ynh_ep_sem32 = ynh_ep_all32.std(axis=0) / np.sqrt(ynh_ep_all32.shape[0])

ynh_ep_mean64 = ynh_ep_all64.mean(axis=0)
ynh_ep_sem64 = ynh_ep_all64.std(axis=0) / np.sqrt(ynh_ep_all64.shape[0])

#%%##MNH 
subjlist_mnh = ['Q351', 'Q363', 'Q364', 'Q365', 'Q368']
            
for subj in range(len(subjlist_mnh)):
    sub = subjlist_mnh[subj]
    dat1 = io.loadmat(data_loc + sub + '_16ms_2-20Hz.mat', squeeze_me=True)
    dat2 = io.loadmat(data_loc + sub + '_32ms_2-20Hz.mat', squeeze_me=True)
    dat3 = io.loadmat(data_loc + sub + '_64ms_2-20Hz.mat', squeeze_me=True)

    dat1.keys()
    dat2.keys()
    dat3.keys()
    
    mnh_ep_mastoid16 = dat1['ep_mastoid']
    mnh_ep_vertex16 = dat1['ep_vertex']
    mnh_ep_ground16= dat1['ep_ground']
    mnh_ep_all16 = dat1['ep_all']
    # mnh_ep_mean16 = dat1['ep_mean']
    # mnh_ep_sem16 = dat1['ep_sem']
    # mnh_ep_subderm16 = dat1['ep_subderm']
    # mnh_ep_mean_subderm16 = dat1['ep_mean_subderm']
    # mnh_ep_sem_subderm16 = dat1['ep_sem_subderm']
    picks= dat1['picks']
    t=dat1['t']
    
    mnh_ep_mastoid32 = dat2['ep_mastoid']
    mnh_ep_vertex32 = dat2['ep_vertex']
    mnh_ep_ground32= dat2['ep_ground']
    mnh_ep_all32 = dat2['ep_all']
    # mnh_ep_mean32 = dat2['ep_mean']
    # mnh_ep_sem32 = dat2['ep_sem']
    # mnh_ep_subderm32 = dat2['ep_subderm']
    # mnh_ep_mean_subderm32 = dat2['ep_mean_subderm']
    # mnh_ep_sem_subderm32 = dat2['ep_sem_subderm']
    
    mnh_ep_mastoid64 = dat3['ep_mastoid']
    mnh_ep_vertex64 = dat3['ep_vertex']
    mnh_ep_ground64= dat3['ep_ground']
    mnh_ep_all64 = dat3['ep_all']
    # mnh_ep_mean64 = dat3['ep_mean']
    # mnh_ep_sem64 = dat3['ep_sem']
    # mnh_ep_subderm64 = dat3['ep_subderm']
    # mnh_ep_mean_subderm64 = dat3['ep_mean_subderm']
    # mnh_ep_sem_subderm64 = dat3['ep_sem_subderm']
    
mnh_vertex16 = mnh_ep_vertex16.mean(axis=0)
mnh_vertex_sem16 = mnh_ep_vertex16.std(axis=0) / np.sqrt(mnh_ep_vertex16.shape[0])
mnh_mastoid16 = mnh_ep_mastoid16.mean(axis=0)
mnh_mastoid_sem16 = mnh_ep_mastoid16.std(axis=0) / np.sqrt(mnh_ep_mastoid16.shape[0])

mnh_vertex32 = mnh_ep_vertex32.mean(axis=0)
mnh_vertex_sem32 = mnh_ep_vertex32.std(axis=0) / np.sqrt(mnh_ep_vertex32.shape[0])
mnh_mastoid32 = mnh_ep_mastoid32.mean(axis=0)
mnh_mastoid_sem32 = mnh_ep_mastoid32.std(axis=0) / np.sqrt(mnh_ep_mastoid32.shape[0])

mnh_vertex64 = mnh_ep_vertex64.mean(axis=0)
mnh_vertex_sem64 = mnh_ep_vertex64.std(axis=0) / np.sqrt(mnh_ep_vertex64.shape[0])
mnh_mastoid64 = mnh_ep_mastoid64.mean(axis=0)
mnh_mastoid_sem64 = mnh_ep_mastoid64.std(axis=0) / np.sqrt(mnh_ep_mastoid64.shape[0])

mnh_ep_mean16 = mnh_ep_all16.mean(axis=0)
mnh_ep_sem16 = mnh_ep_all16.std(axis=0) / np.sqrt(mnh_ep_all16.shape[0])

mnh_ep_mean32 = mnh_ep_all32.mean(axis=0)
mnh_ep_sem32 = mnh_ep_all32.std(axis=0) / np.sqrt(mnh_ep_all32.shape[0])

mnh_ep_mean64 = mnh_ep_all64.mean(axis=0)
mnh_ep_sem64 = mnh_ep_all64.std(axis=0) / np.sqrt(mnh_ep_all64.shape[0])


#%%##TTS
subjlist_tts = ['Q402', 'Q404', 'Q406', 'Q407', 'Q410']

for subj in range(len(subjlist_tts)):
    sub = subjlist_tts[subj]
    dat1 = io.loadmat(data_loc + sub + '_16ms_2-20Hz.mat', squeeze_me=True)
    dat2 = io.loadmat(data_loc + sub + '_32ms_2-20Hz.mat', squeeze_me=True)
    dat3 = io.loadmat(data_loc + sub + '_64ms_2-20Hz.mat', squeeze_me=True)

    dat1.keys()
    dat2.keys()
    dat3.keys()
    
    tts_ep_mastoid16 = dat1['ep_mastoid']
    tts_ep_vertex16 = dat1['ep_vertex']
    tts_ep_ground16= dat1['ep_ground']
    tts_ep_all16 = dat1['ep_all']
    # tts_ep_mean16 = dat1['ep_mean']
    # tts_ep_sem16 = dat1['ep_sem']
    # tts_ep_subderm16 = dat1['ep_subderm']
    # tts_ep_mean_subderm16 = dat1['ep_mean_subderm']
    # tts_ep_sem_subderm16 = dat1['ep_sem_subderm']
    # picks= dat1['picks']
    # t=dat1['t']
    
    tts_ep_mastoid32 = dat2['ep_mastoid']
    tts_ep_vertex32 = dat2['ep_vertex']
    tts_ep_ground32= dat2['ep_ground']
    tts_ep_all32 = dat2['ep_all']
    # tts_ep_mean32 = dat2['ep_mean']
    # tts_ep_sem32 = dat2['ep_sem']
    # tts_ep_subderm32 = dat2['ep_subderm']
    # tts_ep_mean_subderm32 = dat2['ep_mean_subderm']
    # tts_ep_sem_subderm32 = dat2['ep_sem_subderm']
    
    tts_ep_mastoid64 = dat3['ep_mastoid']
    tts_ep_vertex64 = dat3['ep_vertex']
    tts_ep_ground64= dat3['ep_ground']
    tts_ep_all64 = dat3['ep_all']
    # tts_ep_mean64 = dat3['ep_mean']
    # tts_ep_sem64 = dat3['ep_sem']
    # tts_ep_subderm64 = dat3['ep_subderm']
    # tts_ep_mean_subderm64 = dat3['ep_mean_subderm']
    # tts_ep_sem_subderm64 = dat3['ep_sem_subderm']
    
tts_vertex16 = tts_ep_vertex16.mean(axis=0)
tts_vertex_sem16 = tts_ep_vertex16.std(axis=0) / np.sqrt(tts_ep_vertex16.shape[0])
tts_mastoid16 = tts_ep_mastoid16.mean(axis=0)
tts_mastoid_sem16 = tts_ep_mastoid16.std(axis=0) / np.sqrt(tts_ep_mastoid16.shape[0])

tts_vertex32 = tts_ep_vertex32.mean(axis=0)
tts_vertex_sem32 = tts_ep_vertex32.std(axis=0) / np.sqrt(tts_ep_vertex32.shape[0])
tts_mastoid32 = tts_ep_mastoid32.mean(axis=0)
tts_mastoid_sem32 = tts_ep_mastoid32.std(axis=0) / np.sqrt(tts_ep_mastoid32.shape[0])

tts_vertex64 = tts_ep_vertex64.mean(axis=0)
tts_vertex_sem64 = tts_ep_vertex64.std(axis=0) / np.sqrt(tts_ep_vertex64.shape[0])
tts_mastoid64 = tts_ep_mastoid64.mean(axis=0)
tts_mastoid_sem64 = tts_ep_mastoid64.std(axis=0) / np.sqrt(tts_ep_mastoid64.shape[0])

tts_ep_mean16 = tts_ep_all16.mean(axis=0)
tts_ep_sem16 = tts_ep_all16.std(axis=0) / np.sqrt(tts_ep_all16.shape[0])

tts_ep_mean32 = tts_ep_all32.mean(axis=0)
tts_ep_sem32 = tts_ep_all32.std(axis=0) / np.sqrt(tts_ep_all32.shape[0])

tts_ep_mean64 = tts_ep_all64.mean(axis=0)
tts_ep_sem64 = tts_ep_all64.std(axis=0) / np.sqrt(tts_ep_all64.shape[0])

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

