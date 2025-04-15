# -*- coding: utf-8 -*-
"""
Created on Sat Oct 21 13:17:12 2023

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
from scipy.io import savemat
from mne.time_frequency import tfr_multitaper
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
save_loc = 'D:/PhD/Data/Chin_Data/AnalyzedGDT_matfiles/Power_ITC/Induced/'

subjlist = ['Q410',
            'Q428' ]  # Load subject folder

# subjlist = ['Q351', 'Q363', 'Q364', 'Q365', 'Q368',
#             'Q402', 'Q404', 'Q406', 'Q407', 'Q410',
#             'Q412', 'Q422', 'Q424', 'Q426', 'Q428' ]  # Load subject folder

condlist = [1]
gap_dur = 0.032

for subj in subjlist:
    # Load data and read event channel
    fpath = froot + subj + '/'
    bdfs = fnmatch.filter(os.listdir(fpath), subj +'_LightSedation_GDT_32ms*.bdf')
    
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
    # raw.info['bads'].append('A21')
    
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
        raw.info['bads'].append('A31')
        
    if subj == ['Q404']:
        raw.info['bads'].append('A32')
       
# %% Filtering
    raw.filter(0.1, 90.)
    raw.info
    # raw.plot(duration=25.0, n_channels=41, scalings=dict(eeg=100e-6))

    raw.notch_filter(np.arange(60, 241, 60), filter_length='auto', phase='zero')
    
# %% Plotting full responses 

    epochs = mne.Epochs(raw, eves, event_id=[1], baseline=(-0.3, 0), proj=True,tmin=-0.3, tmax=2.2, reject=dict(eeg=200e-6))
    t_full = epochs.times
    evoked = epochs.average()
    
#%%% Power and ITC Analysis 
    freqs = np.arange(0.1, 90., 2.)
    n_cycles = freqs * 0.2
    
    epochs_i = epochs.copy().subtract_evoked()
    
    picks = (6, 7, 8,21, 22, 23, 28, 29, 13)
    power, itc = tfr_multitaper(epochs_i, freqs, n_cycles, picks = picks, 
                                time_bandwidth=4.0, n_jobs=-1, return_itc=True)
    
    # power.plot([0], baseline=(-0.3, 0), mode='mean',title='Gap duration of 16 ms - Power')
    # itc.plot(title='Gap duration of 16 ms - Intertrial Coherence (' + subj + ')',  baseline=(-0.1,0), combine='mean')
   
    # c = power.data.mean (axis=0)
    # plt.plot(freqs, c.mean(axis=1))
    # plt.xlim([0,30])
    # plt.show()
        
    mat_ids = dict (power = power.data, itc=itc.data, picks=picks, freqs=freqs, n_cycles=n_cycles, t_full=t_full)
    savemat(save_loc + subj + '_poweritc_32ms.mat', mat_ids)
    
    print('WOOOHOOOO! Saved ' + subj)
    
    del (epochs, epochs_i, evoked, power, itc)