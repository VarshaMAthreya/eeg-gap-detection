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
from mne.time_frequency import tfr_multitaper
from anlffr.preproc import find_blinks
from mne import compute_proj_epochs
from scipy.io import savemat
from itertools import zip_longest
from scipy.stats import sem 

plt.switch_backend('QT5Agg')  # Making the plots interactive (Scrollable)
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')
# Defining the dimensions and quality of figures
plt.rcParams['figure.dpi'] = 120
plt.rcParams["figure.figsize"] = (5.5, 5)
plt.tight_layout

# %% Loading subjects, reading data, mark bad channels
froot = 'D:/PhD/Data/Chin_Data/Q419_Anes_Awake_GDT+Binding/'  # file location
# save_loc = ('C:/Users/vmysorea/Desktop/PhD/Stim_Analysis/GapDetection_EEG/Analysis/AnalyzedFiles_Figures/YNH/')
# save_loc_ITC = ('C:/Users/vmysorea/Desktop/PhD/Stim_Analysis/GapDetection_EEG/Analysis/AnalyzedFiles_Figures/All_Ages/ITC_matfiles/')

subjlist = ['Q419']  # Load subject folder
condlist = [1, 2, 3]  # List of conditions- Here 3 GDs - 16, 32, 64 ms
condnames = ['0.016 ms', '0.032 ms', '0.064 ms']

for subj in subjlist:
    # evokeds = []
    # itcs = []
    # powers = []
    # Load data and read event channel
    fpath = froot + '/'
    bdfs = fnmatch.filter(os.listdir(fpath), subj +'_Anes_GDT*.bdf')

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
    raw.info['bads'].append('A27') 
    # raw, eves = raw.resample(4096, events=eves)

    # To check and mark bad channels
    # raw.plot(duration=25.0, n_channels=41, scalings=dict(eeg=100e-6))

# %% Filtering
    raw.filter(1., 40.)
    raw.info
    # raw.plot(duration=25.0, n_channels=41, scalings=dict(eeg=100e-6))

# %% Plotting Onset responses
   
    epochs = mne.Epochs(raw, eves, event_id=[1], baseline=(-0.3, 0), proj=True,tmin=-0.3, tmax=2.5, reject=dict(eeg=200e-6))
    t=epochs.times
    evoked = epochs.average()
    # picks = ['A31']
    evoked.plot(titles='Onset - ' + subj)
    
    all_channels = (np.arange(0,31))
    epochs = epochs.get_data()[:, all_channels,:]
    ep_mean = epochs.mean(axis=1)
    evoked = ep_mean.mean(axis=0)
        
    plt.plot(t,evoked, label='Q419 - GDT - Anesthetized')
    plt.show()
    # OnsetResponse_All3.savefig(save_loc + 'OnsetResponse_All3_.png' + subj, dpi=300)

#%% Creating events for each gap (16, 32, 64 ms) 
    
    eves_gaps = eves.copy()
    fs = raw.info['sfreq']
    tone_dur = [0.5, 1.016, 1.548]      ###Hard coded to get 0.5 + gap from the event ID=1 (onset)
    gap_durs = [0.016, 0.032, 0.064]
    events = [2,3,4]
            
    for cond in range(1):
        for a,tone, gap in zip (events,tone_dur, gap_durs):
            event_num = a
            events_add = eves[eves[:,2] == int(cond+1),:] + [(int(fs*tone)+ int(fs*gap)),int(0),event_num - (cond+1)]
            eves_gaps = np.concatenate((eves_gaps,events_add),axis=0)
            
    
#%% Epoching
    
    conds = ['Onset', '16ms', '32ms', '64ms']
    reject=dict(eeg=200e-6)
    
    epochs = []
    evoked = []
    for gaps in range(4):
        ep_gaps = mne.Epochs(raw,eves_gaps,gaps+1,tmin=-0.1,tmax=0.5, reject = reject, baseline =None)
        epochs.append(ep_gaps)
        evoked.append(ep_gaps.average())
        evoked[gaps].plot(picks=11,titles=conds[gaps])
        
    epochs_onset = mne.Epochs(raw, eves_gaps, event_id=[1], baseline=(-0.1, 0), proj=True,tmin=-0.1, tmax=0.5, reject=dict(eeg=200e-6))
    epochs_16 = mne.Epochs(raw,eves_gaps, event_id=[2], baseline=(-0.1, 0), proj=True,tmin=-0.1, tmax=0.5, reject=dict(eeg=200e-6))
    epochs_32 = mne.Epochs(raw, eves_gaps, event_id=[3], baseline=(-0.1, 0), proj=True,tmin=-0.1, tmax=0.5, reject=dict(eeg=200e-6))
    epochs_64 = mne.Epochs(raw, eves_gaps, event_id=[4], baseline=(-0.1, 0), proj=True,tmin=-0.1, tmax=0.5, reject=dict(eeg=200e-6))

# Averaging
    evoked_onset = epochs_onset.average()
    evoked_16 = epochs_16.average()
    evoked_32 = epochs_32.average()
    evoked_64 = epochs_64.average()
    
    # picks = ['A31']
    # evoked_onset.plot(titles='GDT - Onset')
    # evoked_16.plot(titles='GDT_16ms')
    # evoked_32.plot(titles='GDT_32ms')
    # evoked_64.plot(titles='GDT_64ms')
    
    # picks = ['A31', 'A32']
    # evokeds_1 = dict(GDT_16ms=evoked_1, GDT_32ms=evoked_2, GDT_64ms=evoked_3)
    # mne.viz.plot_compare_evokeds(evokeds_1, combine='mean', title='GDT - ' +subj)
    
    t=epochs_16[0].times
    picks = (1, 11, 12, 13, 7,6,22,21)
    all_channels = (np.arange(1,32))
    
    #Onset
    ep_mastoid_onset = epochs_onset.get_data()[:,35,:] #Mastoid -EXG3
    ep_vertex_onset= epochs_onset.get_data()[:,36,:] #Vertex -EXG4
    ep_ground_onset = epochs_onset.get_data()[:,37,:] #Ground - EXG5
    ep_all_onset = evoked_onset.data[picks,:]
    ep_mean_onset =ep_all_onset.mean(axis=0)
    ep_sem_onset = ep_all_onset.std(axis=0) / np.sqrt(ep_all_onset.shape[0])

    ep_subderm_onset = - ep_vertex_onset + ep_mastoid_onset #Inverting mastoid and non-inverting vertex 
    ep_mean_subderm_onset = ep_subderm_onset.mean(axis=0)
    ep_sem_subderm_onset = ep_subderm_onset.std(axis=0) / np.sqrt(ep_subderm_onset.shape[0])

    plt.plot(t, ep_mean_subderm_onset, label='Subdermal electrode')
    plt.fill_between(t, ep_mean_subderm_onset - ep_sem_subderm_onset,
                          ep_mean_subderm_onset + ep_sem_subderm_onset,alpha=0.5)
    plt.plot(t, ep_mean_onset, label = 'EEG Cap - onset ms')
    plt.fill_between(t, ep_mean_onset - ep_sem_onset,
                          ep_mean_onset + ep_sem_onset,alpha=0.5)
    #plt.xticks(ticks= [-2, 0, 2, 4, 6, 8, 10, 12, 14])
    plt.title('GDT-onset ms')
    plt.legend()
    plt.show()
    
    #16 ms
    ep_mastoid_16 = epochs_16.get_data()[:,35,:] #Mastoid -EXG3
    ep_vertex_16= epochs_16.get_data()[:,36,:] #Vertex -EXG4
    ep_ground_16 = epochs_16.get_data()[:,37,:] #Ground - EXG5
    ep_all_16 = evoked_16.data[picks,:]
    ep_mean_16 =ep_all_16.mean(axis=0)
    ep_sem_16 = ep_all_16.std(axis=0) / np.sqrt(ep_all_16.shape[0])

    ep_subderm_16 = - ep_vertex_16 + ep_mastoid_16 #Inverting mastoid and non-inverting vertex 
    ep_mean_subderm_16 = ep_subderm_16.mean(axis=0)
    ep_sem_subderm_16 = ep_subderm_16.std(axis=0) / np.sqrt(ep_subderm_16.shape[0])

    plt.plot(t, ep_mean_subderm_16, label='Subdermal electrode')
    plt.fill_between(t, ep_mean_subderm_16 - ep_sem_subderm_16,
                          ep_mean_subderm_16 + ep_sem_subderm_16,alpha=0.5)
    plt.plot(t, ep_mean_16, label = 'EEG Cap - 16 ms')
    plt.fill_between(t, ep_mean_16 - ep_sem_16,
                          ep_mean_16 + ep_sem_16,alpha=0.5)
    #plt.xticks(ticks= [-2, 0, 2, 4, 6, 8, 10, 12, 14])
    plt.title('GDT-16 ms')
    plt.legend()
    plt.show()
    
    ### 32 ms
    ep_mastoid_32 = epochs_32.get_data()[:,35,:] #Mastoid -EXG3
    ep_vertex_32= epochs_32.get_data()[:,36,:] #Vertex -EXG4
    ep_ground_32 = epochs_32.get_data()[:,37,:] #Ground - EXG5
    ep_all_32 = evoked_32.data[picks,:]
    ep_mean_32 =ep_all_32.mean(axis=0)
    ep_sem_32 = ep_all_32.std(axis=0) / np.sqrt(ep_all_32.shape[0])

    ep_subderm_32 = - ep_vertex_32 + ep_mastoid_32 #Inverting mastoid and non-inverting vertex 
    ep_mean_subderm_32 = ep_subderm_32.mean(axis=0)
    ep_sem_subderm_32 = ep_subderm_32.std(axis=0) / np.sqrt(ep_subderm_32.shape[0])

    plt.plot(t, ep_mean_subderm_32, label='Subdermal electrode')
    plt.fill_between(t, ep_mean_subderm_32 - ep_sem_subderm_32,
                          ep_mean_subderm_32 + ep_sem_subderm_32,alpha=0.5)
    plt.plot(t, ep_mean_32, label = 'EEG Cap-32 ms')
    plt.fill_between(t, ep_mean_32 - ep_sem_32,
                          ep_mean_32 + ep_sem_32,alpha=0.5)
    #plt.xticks(ticks= [-2, 0, 2, 4, 6, 8, 10, 12, 14])
    plt.title('GDT-32 ms')
    plt.legend()
    plt.show()
    
    ### 64 ms 
    ep_mastoid_64 = epochs_64.get_data()[:,35,:] #Mastoid -EXG3
    ep_vertex_64= epochs_64.get_data()[:,36,:] #Vertex -EXG4
    ep_ground_64 = epochs_64.get_data()[:,37,:] #Ground - EXG5
    ep_all_64 = evoked_64.data[picks,:]
    ep_mean_64 =ep_all_64.mean(axis=0)
    ep_sem_64 = ep_all_64.std(axis=0) / np.sqrt(ep_all_64.shape[0])

    ep_subderm_64 = - ep_vertex_64 + ep_mastoid_64 #Inverting mastoid and non-inverting vertex 
    ep_mean_subderm_64 = ep_subderm_64.mean(axis=0)
    ep_sem_subderm_64 = ep_subderm_64.std(axis=0) / np.sqrt(ep_subderm_64.shape[0])

    plt.plot(t, ep_mean_subderm_64, label='Subdermal electrode')
    plt.fill_between(t, ep_mean_subderm_64 - ep_sem_subderm_64,
                          ep_mean_subderm_64 + ep_sem_subderm_64,alpha=0.5)
    plt.plot(t, ep_mean_64, label = 'EEG Cap-64 ms')
    plt.fill_between(t, ep_mean_64 - ep_sem_64,
                          ep_mean_64 + ep_sem_64,alpha=0.5)
    plt.xlim(-0.1, 0.3)
    plt.title('GDT-64 ms')
    plt.legend()
    plt.show()

#%% Plots outside MNE 

    picks = (1, 11, 12, 13, 7,6,22,21)
    combos_comp = [1, 2, 3, 4]
    comp_labels = ['Onset', '16 ms', '32 ms', '64 ms']
    
    fig, ax = plt.subplots(4,1,sharex=True)
    
    t = epochs_16[0].times
    
    for cnd in range(len(combos_comp)):
        cz_12 = (epochs[combos_comp[cnd]].get_data()[:,picks,:]).mean(axis=1)
        cz_mean_12 = cz_12.mean(axis=0)
        cz_sem_12 = sem(cz_12, axis=0)
    
        # cz_20 = (epochs[combos_comp[cnd][1]].get_data()[:,picks,:]).mean(axis=1)
        # cz_mean_20 = cz_20.mean(axis=0)
        # cz_sem_20 = sem(cz_20, axis=0)
        # #cz_sem_20 = cz_ep_20.std(axis=0) / np.sqrt(cz_ep_20.shape[0])
    
        ax[cnd].plot(t,cz_mean_12,label='12')
        ax[cnd].fill_between(t,cz_mean_12 - cz_sem_12, cz_mean_12 + cz_sem_12,alpha=0.5)
    
        # ax[cnd].plot(t,cz_mean_20,label='20')
        # ax[cnd].fill_between(t,cz_mean_20 - cz_sem_20, cz_mean_20 + cz_sem_20,alpha=0.5)
    
        ax[cnd].set_title(comp_labels[cnd])
        ax[cnd].ticklabel_format(axis='y',style='sci',scilimits=(0,0))
    
    # ax[0].legend()
    # ax[2].set_xlabel('Time (sec)')
    # ax[1].set_ylabel('Amplitude (' + u"\u03bcA" + ')')
    # #ax.set_ylim (-5 * 1e-6 , 5 * 1e-6)
    # plt.suptitle(subj + '_Binding')
    plt.show()
# %% Compute evoked response using ITC

# Save location for all analyzed ITC files and figs across ages

    # ITC1 - 16 ms
    freqs = np.arange(1., 20., 1.)
    n_cycles = freqs * 0.2
    t = epochs_16.times
    #epochs_induced = epochs.copy().subtract_evoked()
    # picks = [31]
    power_1, itc_1 = tfr_multitaper(epochs_16, freqs, n_cycles, picks=picks,
                                    time_bandwidth=4.0, n_jobs=-1, return_itc=True)
    #itc_1.apply_baseline(baseline=(-0.5, 0))
    # power_1.apply_baseline(baseline=(-0.3, 0), mode='logratio')
    
    
    # Saving ITC measures into mat file -- Taking the mean across the third row
    x = itc_1.data
    mat_ids1 = dict(itc1=x, freqs=freqs, n_channels=picks, n_cycles=n_cycles, t=t)
    # savemat(save_loc_ITC + 'ITC1_' + subj + '.mat', mat_ids1)
    
    # ITC2 - 32 ms
    freqs = np.arange(1., 20., 1.)
    n_cycles = freqs * 0.2
    t = epochs_32.times
    # picks = [31]
    power_2, itc_2 = tfr_multitaper(epochs_32, freqs, n_cycles, picks=picks,
                                    time_bandwidth=4.0, n_jobs=-1, return_itc=True)
    #itc_2.apply_baseline(baseline=(-0.5, 0))
    # power_2.apply_baseline(baseline=(-0.3, 0), mode='logratio')
    
    y = itc_2.data
    # mat_ids2 = dict(itc2=y, freqs=freqs, n_cycles=n_cycles, t=t)
    # savemat(save_loc_ITC + 'ITC2_' + subj + '.mat', mat_ids2)
    
    # ITC3 - 64 ms
    freqs = np.arange(1., 20., 1.)
    n_cycles = freqs * 0.2
    t = epochs_64.times
    # picks = [31]
    power_3, itc_3 = tfr_multitaper(epochs_64, freqs, n_cycles, picks=picks,
                                    time_bandwidth=4.0, n_jobs=-1, return_itc=True)
    #itc_3.apply_baseline(baseline=(-0.5, 0))
    # power_3.apply_baseline(baseline=(-0.3, 0), mode='logratio')
    
    # Saving ITC measures into mat file -- Taking the mean across the third row
    # z = itc_3.data
    # mat_ids3 = dict(itc3=z, freqs=freqs, n_channels=picks, n_cycles=n_cycles, t=t)
    # savemat(save_loc_ITC + 'ITC3_' + subj + '.mat', mat_ids3)
    
    #itcs += [itc_1, itc_2, itc_3]
    #powers += [power_1, power_2, power_3]
    
    # Plotting ITC and Power plots for the three conditions    
    # Cond 1
    power_1.plot([0], baseline=(-0.3, 0), mode='mean',title='Gap duration of 16 ms - Power')
    itc_1.plot([0], title='Gap duration of 16 ms - Intertrial Coherence (' + subj + ')')
    # #plt.savefig(save_loc + 'ITC1_S105.png', dpi=300)
    
    # # Cond 2
    power_2.plot([0], baseline=(-0.3, 0), mode='mean',title='Gap duration of 32 ms - Power')
    itc_2.plot([0], title='Gap duration of 32 ms- Intertrial Coherence (' + subj + ')')
    # #plt.savefig(save_loc + 'ITC1_S105.png', dpi=300)
    
    # # Cond 3
    power_3.plot([0], baseline=(-0.3, 0), mode='mean', title='Gap duration of 64 ms - Power')
    itc_3.plot([0], title='Gap duration of 64 ms - Intertrial Coherence  (' + subj + ')')
    # #plt.savefig(save_loc + 'ITC1_S105.png', dpi=300)
    
    del epochs, epochs_1, epochs_2,epochs_3,evoked,evoked_1,evoked_2,evoked_3