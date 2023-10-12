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

subjlist = ['Q364']  # Load subject folder
condlist = [1]

for subj in subjlist:
    # Load data and read event channel
    fpath = froot + subj + '/'
    bdfs = fnmatch.filter(os.listdir(fpath), subj +'_LightSedation_ABR.bdf')

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
    raw.info['bads'].append('EXG5') 
    
    raw, eves = raw.resample(32000, events=eves)

    # To check and mark bad channels
    raw.plot(duration=25.0, n_channels=41, scalings=dict(eeg=100e-6))
    
#%% Bad channels for each subject
  
    if subj == ['Q410']:
       raw.info['bads'].append('A12')
       raw.info['bads'].append('A26')
       
    if subj == ['Q428']:
       raw.info['bads'].append('A12')
       raw.info['bads'].append('A26')
       raw.info['bads'].append('A20')
       
    if subj == ['Q422']:
        raw.info['bads'].append('A21')

# %% Filtering
    a = raw.filter(65., 8000., filter_length='10ms', method='iir')
    a.info
    # raw.plot(duration=25.0, n_channels=41, scalings=dict(eeg=100e-6))

# %% Plotting Onset responses

    epochs = mne.Epochs(raw, eves, event_id=[1], baseline=(-0.3, 0), proj=True,tmin=-0.3, tmax=2.5, reject=dict(eeg=200e-6))
    t=epochs.times
    evoked = epochs.average()
    # evoked.plot(titles='Onset - ' + subj)
    
#%% New paradigm with only one gap per trial 
    
    t=epochs[0].times
    picks = (6, 7, 8,21, 22, 23)
    # picks = (1, 11, 12, 13, 7,6,22,21)
    all_channels = (np.arange(1,32))
    
    #Onset
    ep_mastoid = epochs.get_data()[:,35,:] #Mastoid -EXG3
    ep_vertex = epochs.get_data()[:,36,:] #Vertex -EXG4
    ep_ground = epochs.get_data()[:,37,:] #Ground - EXG5
    ep_all = evoked.data[picks,:]
    ep_mean =ep_all.mean(axis=0)
    ep_sem = ep_all.std(axis=0) / np.sqrt(ep_all.shape[0])

    ep_subderm = -ep_vertex + ep_mastoid #Inverting mastoid and non-inverting vertex 
    ep_mean_subderm = ep_subderm.mean(axis=0)
    ep_sem_subderm = ep_subderm.std(axis=0) / np.sqrt(ep_subderm.shape[0])

    # plt.plot(t, ep_mean_subderm, label='Subdermal electrode')
    # plt.fill_between(t, ep_mean_subderm - ep_sem_subderm,
    #                       ep_mean_subderm + ep_sem_subderm,alpha=0.5)
    plt.plot(t, ep_mean, label = 'EEG Cap' + str(picks))
    plt.fill_between(t, ep_mean - ep_sem,
                          ep_mean + ep_sem,alpha=0.5)
    plt.xlim(-0.1, 2.1)
    #plt.xticks(ticks= [-2, 0, 2, 4, 6, 8, 10, 12, 14])
    plt.title('Q428 (YNH) - Light Sedation : GDT- 64 ms')
    plt.legend()
    plt.show()
    
    # plt.savefig(save_loc + 'Q410_16ms.png', dpi=300)
    
    ### Saving mat files
       
    mat_ids = dict(ep_mastoid = ep_mastoid, ep_vertex = ep_vertex, ep_ground=ep_ground, ep_all = ep_all, ep_mean =ep_mean, 
                    ep_sem = ep_sem, ep_subderm = ep_subderm, ep_mean_subderm = ep_mean_subderm, ep_sem_subderm = ep_sem_subderm,
                    picks=picks, t=t) 
    savemat(save_loc + subj + '_32ms.mat', mat_ids)

#%% Creating events for each gap (16, 32, 64 ms) -- When each trial is tone +gap1+tone+gap2+tone+gap3
    
    # eves_gaps = eves.copy()
    # fs = raw.info['sfreq']
    # tone_dur = [0.5, 1.016, 1.548]      ###Hard coded to get 0.5 + gap from the event ID=1 (onset)
    # gap_durs = [0.016, 0.032, 0.064]
    # events = [2,3,4]
            
    # for cond in range(1):
    #     for a,tone, gap in zip (events,tone_dur, gap_durs):
    #         event_num = a
    #         events_add = eves[eves[:,2] == int(cond+1),:] + [(int(fs*tone)+ int(fs*gap)),int(0),event_num - (cond+1)]
    #         eves_gaps = np.concatenate((eves_gaps,events_add),axis=0)

#%% Creating events for each gap (16, 32, 64 ms)      - When each trial is tone + gap + tone        
    # epochs_manual = mne.Epochs(raw, eves, event_id=[1], baseline=(1.3, 1.5), proj=True,tmin=1.3, tmax=2.5, reject=dict(eeg=200e-6))
    # t1=epochs_manual.times
    # evoked_manual = epochs_manual.average()
    # # picks = ['A31']
    # evoked_manual.plot(titles='Onset - ' + subj)
    
    
    # t1=epochs_manual[0].times
    # picks = (6, 7, 8, 21, 22, 23)
    # # picks = (1, 11, 12, 13, 7,6,22,21)
    # all_channels = (np.arange(1,32))
    
    # #Onset
    # ep_mastoid = epochs_manual.get_data()[:,35,:] #Mastoid -EXG3
    # ep_vertex = epochs_manual.get_data()[:,36,:] #Vertex -EXG4
    # ep_ground = epochs_manual.get_data()[:,37,:] #Ground - EXG5
    # ep_all = evoked_manual.data[picks,:]
    # ep_mean =ep_all.mean(axis=0)
    # ep_sem = ep_all.std(axis=0) / np.sqrt(ep_all.shape[0])

    # ep_subderm = ep_vertex - ep_mastoid #Inverting mastoid and non-inverting vertex 
    # ep_mean_subderm = ep_subderm.mean(axis=0)
    # ep_sem_subderm = ep_subderm.std(axis=0) / np.sqrt(ep_subderm.shape[0])

    # plt.plot(t1, ep_mean_subderm, label='Subdermal electrode')
    # plt.fill_between(t1, ep_mean_subderm - ep_sem_subderm,
    #                       ep_mean_subderm + ep_sem_subderm,alpha=0.5)
    # plt.plot(t1, ep_mean, label = 'EEG Cap' + str(picks))
    # plt.fill_between(t1, ep_mean - ep_sem,
    #                       ep_mean + ep_sem,alpha=0.5)
    # plt.xlim(1.5, 2.1)
    # #plt.xticks(ticks= [-2, 0, 2, 4, 6, 8, 10, 12, 14])
    # plt.title('Q428 (YNH) - Light Sedation : GDT- 64 ms')
    # plt.legend()
    # plt.show()
    
            
    
# #%% Epoching
    
#     conds = ['Onset', '16ms', '32ms', '64ms']
#     reject=dict(eeg=200e-6)
    
#     epochs = []
#     evoked = []
#     for gaps in range(4):
#         ep_gaps = mne.Epochs(raw,eves_gaps,gaps+1,tmin=-0.1,tmax=0.5, reject = reject, baseline =None)
#         epochs.append(ep_gaps)
#         evoked.append(ep_gaps.average())
#         evoked[gaps].plot(picks=11,titles=conds[gaps])
        
#     epochs_onset = mne.Epochs(raw, eves_gaps, event_id=[1], baseline=(-0.1, 0), proj=True,tmin=-0.1, tmax=0.5, reject=dict(eeg=200e-6))
#     epochs_16 = mne.Epochs(raw,eves_gaps, event_id=[2], baseline=(-0.1, 0), proj=True,tmin=-0.1, tmax=0.5, reject=dict(eeg=200e-6))
#     epochs_32 = mne.Epochs(raw, eves_gaps, event_id=[3], baseline=(-0.1, 0), proj=True,tmin=-0.1, tmax=0.5, reject=dict(eeg=200e-6))
#     epochs_64 = mne.Epochs(raw, eves_gaps, event_id=[4], baseline=(-0.1, 0), proj=True,tmin=-0.1, tmax=0.5, reject=dict(eeg=200e-6))

# # Averaging
#     evoked_onset = epochs_onset.average()
#     evoked_16 = epochs_16.average()
#     evoked_32 = epochs_32.average()
#     evoked_64 = epochs_64.average()
    
#     # picks = ['A31']
#     # evoked_onset.plot(titles='GDT - Onset')
#     # evoked_16.plot(titles='GDT_16ms')
#     # evoked_32.plot(titles='GDT_32ms')
#     # evoked_64.plot(titles='GDT_64ms')
    
#     # picks = ['A31', 'A32']
#     # evokeds_1 = dict(GDT_16ms=evoked_1, GDT_32ms=evoked_2, GDT_64ms=evoked_3)
#     # mne.viz.plot_compare_evokeds(evokeds_1, combine='mean', title='GDT - ' +subj)
    
#     t=epochs_16[0].times
#     picks = [1, 11, 12, 13, 7,6,22,21]
#     all_channels = (np.arange(1,32))
    
#     #Onset
#     ep_mastoid_onset = epochs_onset.get_data()[:,35,:] #Mastoid -EXG3
#     ep_vertex_onset= epochs_onset.get_data()[:,36,:] #Vertex -EXG4
#     ep_ground_onset = epochs_onset.get_data()[:,37,:] #Ground - EXG5
#     ep_all_onset = evoked_onset.data[picks,:]
#     ep_mean_onset =ep_all_onset.mean(axis=0)
#     ep_sem_onset = ep_all_onset.std(axis=0) / np.sqrt(ep_all_onset.shape[0])

#     ep_subderm_onset = - ep_vertex_onset + ep_mastoid_onset #Inverting mastoid and non-inverting vertex 
#     ep_mean_subderm_onset = ep_subderm_onset.mean(axis=0)
#     ep_sem_subderm_onset = ep_subderm_onset.std(axis=0) / np.sqrt(ep_subderm_onset.shape[0])

#     plt.plot(t, ep_mean_subderm_onset, label='Subdermal electrode')
#     plt.fill_between(t, ep_mean_subderm_onset - ep_sem_subderm_onset,
#                           ep_mean_subderm_onset + ep_sem_subderm_onset,alpha=0.5)
#     plt.plot(t, ep_mean_onset, label = 'EEG Cap - onset ms')
#     plt.fill_between(t, ep_mean_onset - ep_sem_onset,
#                           ep_mean_onset + ep_sem_onset,alpha=0.5)
#     #plt.xticks(ticks= [-2, 0, 2, 4, 6, 8, 10, 12, 14])
#     plt.title('GDT-onset ms')
#     plt.legend()
#     plt.show()
    
#     #16 ms
#     ep_mastoid_16 = epochs_16.get_data()[:,35,:] #Mastoid -EXG3
#     ep_vertex_16= epochs_16.get_data()[:,36,:] #Vertex -EXG4
#     ep_ground_16 = epochs_16.get_data()[:,37,:] #Ground - EXG5
#     ep_all_16 = evoked_16.data[picks,:]
#     ep_mean_16 =ep_all_16.mean(axis=0)
#     ep_sem_16 = ep_all_16.std(axis=0) / np.sqrt(ep_all_16.shape[0])

#     ep_subderm_16 = - ep_vertex_16 + ep_mastoid_16 #Inverting mastoid and non-inverting vertex 
#     ep_mean_subderm_16 = ep_subderm_16.mean(axis=0)
#     ep_sem_subderm_16 = ep_subderm_16.std(axis=0) / np.sqrt(ep_subderm_16.shape[0])

#     plt.plot(t, ep_mean_subderm_16, label='Subdermal electrode')
#     plt.fill_between(t, ep_mean_subderm_16 - ep_sem_subderm_16,
#                           ep_mean_subderm_16 + ep_sem_subderm_16,alpha=0.5)
#     plt.plot(t, ep_mean_16, label = 'EEG Cap - 16 ms')
#     plt.fill_between(t, ep_mean_16 - ep_sem_16,
#                           ep_mean_16 + ep_sem_16,alpha=0.5)
#     #plt.xticks(ticks= [-2, 0, 2, 4, 6, 8, 10, 12, 14])
#     plt.title('GDT-16 ms')
#     plt.legend()
#     plt.show()
    
#     ### 32 ms
#     ep_mastoid_32 = epochs_32.get_data()[:,35,:] #Mastoid -EXG3
#     ep_vertex_32= epochs_32.get_data()[:,36,:] #Vertex -EXG4
#     ep_ground_32 = epochs_32.get_data()[:,37,:] #Ground - EXG5
#     ep_all_32 = evoked_32.data[picks,:]
#     ep_mean_32 =ep_all_32.mean(axis=0)
#     ep_sem_32 = ep_all_32.std(axis=0) / np.sqrt(ep_all_32.shape[0])

#     ep_subderm_32 = - ep_vertex_32 + ep_mastoid_32 #Inverting mastoid and non-inverting vertex 
#     ep_mean_subderm_32 = ep_subderm_32.mean(axis=0)
#     ep_sem_subderm_32 = ep_subderm_32.std(axis=0) / np.sqrt(ep_subderm_32.shape[0])

#     plt.plot(t, ep_mean_subderm_32, label='Subdermal electrode')
#     plt.fill_between(t, ep_mean_subderm_32 - ep_sem_subderm_32,
#                           ep_mean_subderm_32 + ep_sem_subderm_32,alpha=0.5)
#     plt.plot(t, ep_mean_32, label = 'EEG Cap-32 ms')
#     plt.fill_between(t, ep_mean_32 - ep_sem_32,
#                           ep_mean_32 + ep_sem_32,alpha=0.5)
#     #plt.xticks(ticks= [-2, 0, 2, 4, 6, 8, 10, 12, 14])
#     plt.title('GDT-32 ms')
#     plt.legend()
#     plt.show()
    
#     ### 64 ms 
#     ep_mastoid_64 = epochs_64.get_data()[:,35,:] #Mastoid -EXG3
#     ep_vertex_64= epochs_64.get_data()[:,36,:] #Vertex -EXG4
#     ep_ground_64 = epochs_64.get_data()[:,37,:] #Ground - EXG5
#     ep_all_64 = evoked_64.data[picks,:]
#     ep_mean_64 =ep_all_64.mean(axis=0)
#     ep_sem_64 = ep_all_64.std(axis=0) / np.sqrt(ep_all_64.shape[0])

#     ep_subderm_64 = - ep_vertex_64 + ep_mastoid_64 #Inverting mastoid and non-inverting vertex 
#     ep_mean_subderm_64 = ep_subderm_64.mean(axis=0)
#     ep_sem_subderm_64 = ep_subderm_64.std(axis=0) / np.sqrt(ep_subderm_64.shape[0])

#     plt.plot(t, ep_mean_subderm_64, label='Subdermal electrode')
#     plt.fill_between(t, ep_mean_subderm_64 - ep_sem_subderm_64,
#                           ep_mean_subderm_64 + ep_sem_subderm_64,alpha=0.5)
#     plt.plot(t, ep_mean_64, label = 'EEG Cap-64 ms')
#     plt.fill_between(t, ep_mean_64 - ep_sem_64,
#                           ep_mean_64 + ep_sem_64,alpha=0.5)
#     plt.xlim(-0.1, 0.3)
#     plt.title('GDT-64 ms')
#     plt.legend()
#     plt.show()


