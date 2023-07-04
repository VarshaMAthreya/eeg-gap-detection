# -*- coding: utf-8 -*-
"""
Created on Sun Nov  7 18:27:02 2021

@author: vmysorea
"""
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

plt.switch_backend('QT5Agg')  # Making the plots interactive (Scrollable)
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')
# Defining the dimensions and quality of figures
plt.rcParams['figure.dpi'] = 120
plt.rcParams["figure.figsize"] = (5.5, 5)
plt.tight_layout

# %% Loading subjects, reading data, mark bad channels
froot = 'D:/PhD/Data/MTB_EP - GDT, Binding, mTRF/GDT/'  # file location
save_loc = ('C:/Users/vmysorea/Desktop/PhD/Stim_Analysis/GapDetection_EEG/Analysis/AnalyzedFiles_Figures/YNH/')
save_loc_ITC = ('C:/Users/vmysorea/Desktop/PhD/Stim_Analysis/GapDetection_EEG/Analysis/AnalyzedFiles_Figures/All_Ages/ITC_matfiles/')

subjlist = ['S268']  # Load subject folder
condlist = [1, 2, 3]  # List of conditions- Here 3 GDs - 16, 32, 64 ms
condnames = ['0.016 ms', '0.032 ms', '0.064 ms']

for subj in subjlist:
    evokeds = []
    
    # Load data and read event channel
    fpath = froot + subj + '/'
    bdfs = fnmatch.filter(os.listdir(fpath), subj +'_GDT*.bdf')

    # Load data and read event channel
    rawlist = []
    evelist = []

    for k, rawname in enumerate(bdfs):
        rawtemp, evestemp = bs.importbdf(fpath + rawname, verbose='DEBUG',refchans=['EXG1', 'EXG2'])
        rawlist += [rawtemp, ]
        evelist += [evestemp, ]
    raw, eves = mne.concatenate_raws(rawlist, events_list=evelist)
    # raw, eves = raw.resample(4096, events=eves)

    # To check and mark bad channels
    raw.plot(duration=25.0, n_channels=32, scalings=dict(eeg=100e-6))

#%% Reject a few of the electrodes for each subject
  
    if subj == 'S273':
       raw.info['bads'].append('A24')
       raw.info['bads'].append('A7')
      
    if subj == 'S268':
       raw.info['bads'].append('A25')
       raw.info['bads'].append('A29')
    
    if subj == 'S288':
       raw.info['bads'].append('A27')
       raw.info['bads'].append('A28')
       raw.info['bads'].append('A30')
       raw.info['bads'].append('A1')
        

# %% Filtering
    raw.filter(1., 40.)
    raw.info

# %% Blink Rejection
    blinks = find_blinks(raw)
    raw.plot(events=blinks, duration=25.0, n_channels=32, scalings=dict(eeg=200e-6))
    epochs_blinks = mne.Epochs(raw, blinks, event_id=998, baseline=(-0.25, 0.25),reject=dict(eeg=500e-6), tmin=-0.25, tmax=0.25)
    blink_proj = compute_proj_epochs(epochs_blinks, n_eeg=1)
    raw.add_proj(blink_proj)  # Adding the n=blink proj to the data -- removal of blink
    raw.plot_projs_topomap()  # Visualizing the spatial filter

# %% Plotting Onset responses -- Remember, the trigs 1,2,3 correspond to the beginning of a trial containing 3 gaps of 0.016, 0.032, 0.064 ms respectively
    for c, cond in enumerate(condlist):
    
        condname = condnames[c]
        epochs = mne.Epochs(raw, eves, cond, tmin=-0.3, proj=True,tmax=2.5, baseline=(-0.3, 0.0),reject=dict(eeg=200e-6))
        evoked = epochs.average()
        evokeds += [evoked, ]
        picks = ['A31']
        evoked.plot(picks=picks, titles='Onset Response' + condname)
    
    epochs = mne.Epochs(raw, eves, event_id=[1, 2, 3], baseline=(-0.3, 0), proj=True,tmin=-0.3, tmax=2.2, reject=dict(eeg=200e-6))
    evoked = epochs.average()
    picks = ['A31']
    OnsetResponse_All3 = evoked.plot(picks=picks, titles='Onset - ' + subj)
    #ylim=dict(eeg=[-1.1, 1.5])
    # OnsetResponse_All3.savefig(save_loc + 'OnsetResponse_All3_.png' + subj, dpi=300)

# %% Creating manual events to add the responses of each gap size
    
    fs = raw.info['sfreq']
    gap_durs = [.016, .032, 0.064]
    eves_manual = np.zeros((3*eves.shape[0], 3))
    for k in range(1, eves.shape[0]):
        for m in range(3):
            current_eves = eves[k, :].copy()
            gap_samps = gap_durs[int(current_eves[2])-1] * fs
            current_eves[0] = current_eves[0] + ((m + 1) * np.round(0.5*fs) + m*gap_samps)  #0.5 is the tone duration (No need for ISI because OG trigs would have taken care of that)
            eves_manual[((k-1)*3) + m, :] = current_eves
    
    eves_manual = np.int64(eves_manual)
    
    # Epoching
    
    epochs_1 = mne.Epochs(raw, eves_manual[:], event_id=[1], baseline=(-0.1, 0), proj=True,tmin=-0.1, tmax=0.6, reject=dict(eeg=200e-6))
    epochs_2 = mne.Epochs(raw, eves_manual, event_id=[2], baseline=(-0.1, 0), proj=True,tmin=-0.1, tmax=0.6, reject=dict(eeg=200e-6))
    epochs_3 = mne.Epochs(raw, eves_manual, event_id=[3], baseline=(-0.1, 0), proj=True,tmin=-0.1, tmax=0.6, reject=dict(eeg=200e-6))

# Averaging
    evoked_1 = epochs_1.average()
    evoked_2 = epochs_2.average()
    evoked_3 = epochs_3.average()
    
    picks = ['A32']
    evoked_1.plot(ylim=dict(eeg=[-1.1, 1.5]), titles='GDT_16ms - ' + subj)
    evoked_2.plot(ylim=dict(eeg=[-1.1, 1.5]), titles='GDT_32ms - ' + subj)
    evoked_3.plot(ylim=dict(eeg=[-1.1, 1.5]), titles='GDT_64ms - ' + subj)
    
    picks = ['A32']
    evokeds = dict(GDT_16ms=evoked_1, GDT_32ms=evoked_2, GDT_64ms=evoked_3)
    mne.viz.plot_compare_evokeds(evokeds, combine='mean', title='GDT - ' +subj, picks=picks)

# %% Compute evoked response using ITC

# Save location for all analyzed ITC files and figs across ages

    # ITC1 - 16 ms
    freqs = np.arange(1., 14., 1.)
    n_cycles = freqs * 0.2
    t = epochs_1.times
    #epochs_induced = epochs.copy().subtract_evoked()
    picks = [31]
    power_1, itc_1 = tfr_multitaper(epochs_1, freqs, n_cycles, picks=picks,
                                    time_bandwidth=4.0, n_jobs=-1, return_itc=True)
    #itc_1.apply_baseline(baseline=(-0.5, 0))
    # power_1.apply_baseline(baseline=(-0.3, 0), mode='logratio')
    
    
    # Saving ITC measures into mat file -- Taking the mean across the third row
    x = itc_1.data
    mat_ids1 = dict(itc1=x, freqs=freqs, n_channels=picks, n_cycles=n_cycles, t=t)
    savemat(save_loc_ITC + 'ITC1_' + subj + '.mat', mat_ids1)
    
    # ITC2 - 32 ms
    freqs = np.arange(1., 14., 1.)
    n_cycles = freqs * 0.2
    t = epochs_2.times
    picks = [31]
    power_2, itc_2 = tfr_multitaper(epochs_2, freqs, n_cycles, picks=picks,
                                    time_bandwidth=4.0, n_jobs=-1, return_itc=True)
    #itc_2.apply_baseline(baseline=(-0.5, 0))
    # power_2.apply_baseline(baseline=(-0.3, 0), mode='logratio')
    
    y = itc_2.data
    mat_ids2 = dict(itc2=y, freqs=freqs, n_cycles=n_cycles, t=t)
    savemat(save_loc_ITC + 'ITC2_' + subj + '.mat', mat_ids2)
    
    # ITC3 - 64 ms
    freqs = np.arange(1., 14., 1.)
    n_cycles = freqs * 0.2
    t = epochs_3.times
    picks = [31]
    power_3, itc_3 = tfr_multitaper(epochs_3, freqs, n_cycles, picks=picks,
                                    time_bandwidth=4.0, n_jobs=-1, return_itc=True)
    #itc_3.apply_baseline(baseline=(-0.5, 0))
    # power_3.apply_baseline(baseline=(-0.3, 0), mode='logratio')
    
    # Saving ITC measures into mat file -- Taking the mean across the third row
    z = itc_3.data
    mat_ids3 = dict(itc3=z, freqs=freqs, n_channels=picks, n_cycles=n_cycles, t=t)
    savemat(save_loc_ITC + 'ITC3_' + subj + '.mat', mat_ids3)
    
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