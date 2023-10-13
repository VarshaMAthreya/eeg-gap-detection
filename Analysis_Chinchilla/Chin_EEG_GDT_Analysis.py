# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 14:46:03 2023

@author: vmysorea
"""
### Chin GDT analysis 

import sys
sys.path.append('C:/Users/vmysorea/Documents/mne-python/')
sys.path.append('C:/Users/vmysorea/Documents/ANLffr/')
import warnings
import mne
import numpy as np
from anlffr.helper import biosemi2mne as bs
from matplotlib import pyplot as plt
import os
import fnmatch
from scipy.io import savemat
# from itertools import zip_longest
# from scipy.stats import sem 

plt.switch_backend('QT5Agg')  # Making the plots interactive (Scrollable)
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')
# Defining the dimensions and quality of figures
plt.rcParams['figure.dpi'] = 120
plt.rcParams["figure.figsize"] = (5.5, 5)
plt.tight_layout

# %% Loading subjects, reading data, mark bad channels
froot = 'D:/PhD/Data/Chin_Data/LightSedation/'  # file location
save_loc = 'D:/PhD/Data/Chin_Data/AnalyzedGDT_matfiles/'

# subjlist = ['Q426']

subjlist = ['Q351', 'Q363', 'Q364', 'Q365', 'Q368',
            'Q402', 'Q404', 'Q406', 'Q407', 'Q410',
            'Q412', 'Q422', 'Q424', 'Q426', 'Q428' ]  # Load subject folder

condlist = [1]
gap_dur = 0.064

for subj in subjlist:
    # Load data and read event channel
    fpath = froot + subj + '/'
    bdfs = fnmatch.filter(os.listdir(fpath), subj +'_LightSedation_GDT_64ms*.bdf')
    
    print('LOADING! ' + subj +' raw data')

    # Load data and read event channel
    rawlist = []
    evelist = []

    for k, rawname in enumerate(bdfs):
        rawtemp, evestemp = bs.importbdf(fpath + rawname, verbose='DEBUG',refchans=['EXG1', 'EXG2'])
        rawlist += [rawtemp, ]
        evelist += [evestemp, ]
    raw, eves = mne.concatenate_raws(rawlist, events_list=evelist)
    raw.set_channel_types({'EXG3':'eeg'})           #Mastoid -34
    raw.set_channel_types({'EXG4':'eeg'})           #Vertex -35
    raw.set_channel_types({'EXG5':'eeg'})           #Ground -36
    raw.info['bads'].append('EXG6') 
    raw.info['bads'].append('EXG7')
    raw.info['bads'].append('EXG8')
    raw.info['bads'].append('A12') 
    raw.info['bads'].append('A26') 
    raw.info['bads'].append('A27') 
    raw.info['bads'].append('A20') 
    raw.info['bads'].append('A17')
    # raw.info['bads'].append('EXG5')
    
    raw, eves = raw.resample(4096, events=eves)

    # To check and mark bad channels
    # raw.plot(duration=25.0, n_channels=41, scalings=dict(eeg=100e-6))
    
#%% Bad channels for each subject
  
    if subj == ['Q364']:
       raw.info['bads'].append('A13')
       
    if subj == ['Q404']:
       raw.info['bads'].append('A13')
       
    if subj == ['Q412', 'Q424']:
       raw.info['bads'].append('A1')
       
    if subj == ['Q422']:
        raw.info['bads'].append('EXG5')
        raw.info['bads'].append('A21')
        
    if subj == ['Q426']:
        raw.info['bads'].append('A5')
       
# %% Filtering
    raw.filter(2., 20.)
    raw.info
    # raw.plot(duration=25.0, n_channels=41, scalings=dict(eeg=100e-6))

# %% Plotting full responses 

    epochs = mne.Epochs(raw, eves, event_id=[1], baseline=(-0.3, 0), proj=True,tmin=-0.3, tmax=2.2, reject=dict(eeg=200e-6))
    t_full = epochs.times
    evoked = epochs.average()
    # evoked.plot(titles='Onset - ' + subj)
    
#%% Manual plots and saving 
    
    all_channels = (np.arange(1,32))
    # picks = all_channels
    picks = (6, 7, 8,21, 22, 23, 28, 29, 13)
    
    #Onset
    ep_mastoid = (epochs.get_data()[:,35,:]).mean(axis=0) #Mastoid -EXG3
    ep_vertex = (epochs.get_data()[:,36,:]).mean(axis=0) #Vertex -EXG4
    ep_ground = (epochs.get_data()[:,37,:]).mean(axis=0) #Ground - EXG5
    ep_all = evoked.data[picks,:]
    # ep_mean =ep_all.mean(axis=0)
    # ep_sem = ep_all.std(axis=0) / np.sqrt(ep_all.shape[0])

    # ep_subderm = -ep_vertex + ep_mastoid #Inverting mastoid and non-inverting vertex 
    # ep_mean_subderm = ep_subderm.mean(axis=0)
    # ep_sem_subderm = ep_subderm.std(axis=0) / np.sqrt(ep_subderm.shape[0])

    # plt.plot(t, ep_mean_subderm, label='Subdermal electrode')
    # plt.fill_between(t, ep_mean_subderm - ep_sem_subderm,
    #                       ep_mean_subderm + ep_sem_subderm,alpha=0.5)
    # plt.plot(t_full, ep_mean, label = 'EEG Cap' + str(picks))
    # plt.fill_between(t_full, ep_mean - ep_sem,
    #                       ep_mean + ep_sem,alpha=0.5)
    # plt.xlim(-0.1, 2.1)
    #plt.xticks(ticks= [-2, 0, 2, 4, 6, 8, 10, 12, 14])
    # plt.title('Q428 (YNH) - Light Sedation : GDT- 64 ms')
    # plt.legend()
    # plt.show()
        
    ### Saving mat files
       
    mat_all = dict(ep_mastoid = ep_mastoid, ep_vertex = ep_vertex, ep_ground=ep_ground, ep_all = ep_all,t_full=t_full) 

#%% Creating events for each gap separately (16, 32, 64 ms) -- When each trial is tone + gap + tone 
    
    eves_gaps = eves.copy()
    fs = raw.info['sfreq']
    tone_dur = 1      ###Hard coded to get 1 s + gap from the event ID=1 (onset)
    event_num = 2
            
    for cond in range(1):
        event_num = event_num
        events_add = eves[eves[:,2] == int(cond+1),:] + [(int(fs*tone_dur)+ int(fs*gap_dur)),int(0),event_num - (cond+1)]
        eves_gaps = np.concatenate((eves_gaps,events_add),axis=0)
                
    epoch_gap = mne.Epochs(raw, eves_gaps, event_id=[2], baseline=(-0.2,-0.07), proj=True,tmin=-0.2, tmax=1.2, reject=dict(eeg=200e-6))
    t=epoch_gap.times
    evoked_gap = epoch_gap.average()
    # evoked_gap.plot()
    
#%%% Plotting gap responses 
    
    # picks = all_channels 
    
    gap_mastoid = (epoch_gap.get_data()[:,35,:]).mean(axis=0) #Mastoid -EXG3
    gap_vertex = (epoch_gap.get_data()[:,36,:]).mean(axis=0) #Vertex -EXG4
    gap_ground = (epoch_gap.get_data()[:,37,:]).mean(axis=0) #Ground - EXG5
    gap_cap = evoked_gap.data[picks,:]
    gap_cap_mean =gap_cap.mean(axis=0)
    # gap_sem = gap_cap.std(axis=0) / np.sqrt(gap_cap.shape[0])

    # gap_subderm = gap_vertex - gap_mastoid #Inverting mastoid and non-inverting vertex 
    # gap_mean_subderm = gap_subderm.mean(axis=0)
    # gap_sem_subderm = gap_subderm.std(axis=0) / np.sqrt(gap_subderm.shape[0])

    # plt.plot(t, gap_mean_subderm, label='Subdermal electrode')
    # plt.fill_between(t, gap_mean_subderm - gap_sem_subderm,
    #                       gap_mean_subderm + gap_sem_subderm,alpha=0.5)
    
    # plt.plot(t, gap_cap_mean, label = 'EEG Cap' + str(picks))
    # plt.fill_between(t, gap_cap_mean - gap_sem,
    #                       gap_cap_mean + gap_sem,alpha=0.5)
    # # plt.xlim(1.5, 2.1)
    # #plt.xticks(ticks= [-2, 0, 2, 4, 6, 8, 10, 12, 14])
    # plt.title('Q428 (YNH) - Light Sedation : GDT- 64 ms')
    # plt.legend()
    # plt.show()
    
    mat_gap = dict(gap_mastoid = gap_mastoid, gap_vertex = gap_vertex, gap_ground=gap_ground, gap_cap = gap_cap, t=t, picks=picks) 
    
    mat_ids = mat_all | mat_gap
    
    savemat(save_loc + subj + '_64ms_2-20Hz.mat', mat_ids)
    
    print('WOOOHOOOO! Saved ' + subj)
   
    del (epochs, epoch_gap, evoked, evoked_gap)