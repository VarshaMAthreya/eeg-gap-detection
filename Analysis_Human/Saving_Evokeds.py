# -*- coding: utf-8 -*-
"""
Created on Sat Feb  4 23:31:07 2023

@author: vmysorea
"""

#Saving epochs for full time
import sys
sys.path.append('C:/Users/vmysorea/Documents/mne-python/')
sys.path.append('C:/Users/vmysorea/Documents/ANLffr/')
import warnings
import mne
from anlffr.helper import biosemi2mne as bs
from matplotlib import pyplot as plt
import os
import fnmatch
from anlffr.preproc import find_blinks
from mne import compute_proj_epochs
# import numpy as np
from scipy.io import savemat

plt.switch_backend('QT5Agg')  # Making the plots interactive (Scrollable)
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')
# %% Loading subjects, reading data, mark bad channels
froot = 'D:/PhD/Data/MTB_EP - GDT, Binding, mTRF/GDT/'  # file location
save_mat_loc = 'D:/PhD/Data/GDT_matfiles/'

subjlist = ['S273','S268','S269','S274','S282','S285','S259','S277','S279','S280','S270','S271','S281','S290','S284',
              'S305','S303','S288','S260','S341','S312','S347','S340','S078','S069', 'S088','S072','S308',
              'S344','S105','S291','S310','S339']
            
condlist = [1, 2, 3]  # List of conditions- Here 3 GDs - 16, 32, 64 ms
condnames = ['0.016 ms', '0.032 ms', '0.064 ms']

for subj in subjlist:
    
    fpath = froot + subj + '/'
    bdfs = fnmatch.filter(os.listdir(fpath), subj +'_GDT*.bdf')      # Load data and read event channel
    print('LOADING! ' + subj +' raw data')

    rawlist = []
    evelist = []

    for k, rawname in enumerate(bdfs):
        rawtemp, evestemp = bs.importbdf(fpath + rawname, verbose='DEBUG', refchans=['EXG1', 'EXG2'])
        rawlist += [rawtemp, ]
        evelist += [evestemp, ]
    raw, eves = mne.concatenate_raws(rawlist, events_list=evelist)
    #raw.plot(duration=25.0, n_channels=32, scalings=dict(eeg=100e-6), event_color={1: 'r', 2: 'g'})    # To check and mark bad channels
    
    #%% Filtering
    raw.filter(1, 40.)
    # raw.info

# %% Blink Rejection
    blinks = find_blinks(raw)
    #raw.plot(events=blinks, duration=25.0, n_channels=32, scalings=dict(eeg=200e-6))
    epochs_blinks = mne.Epochs(raw, blinks, event_id=998, baseline=(-0.25, 0.25), reject=dict(eeg=500e-6), tmin=-0.25, tmax=0.25)
    blink_proj = compute_proj_epochs(epochs_blinks, n_eeg=1)
    raw.add_proj(blink_proj)  # Adding the n=blink proj to the data -- removal of blinks
    #raw.plot_projs_topomap()     # Visualizing the spatial filter

#%% Saving epochs and evoked to fiff files 
    picks=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31]
    
    epochs = mne.Epochs(raw, eves, event_id=[1, 2, 3], baseline=(-0.3, 0), proj=True,tmin=-0.3, tmax=2.5, 
                        reject=dict(eeg=200e-6),picks=picks)
    evoked = epochs.average(picks=picks)
    # epochs.save(save_epochs_loc + subj + '_GDT_epochs-epo.fif', fmt = 'double', overwrite=True)
    # evoked.save(save_epochs_loc + subj + '_GDT_evoked-ave.fif', overwrite=True)
    
    # epochs1 = mne.Epochs(raw, eves, event_id=[1],  proj=True, baseline = (-0.3,0), tmin=-0.3, tmax=5.2, 
    #                      reject=dict(eeg=150e-6), picks=picks)
    # evoked1 = epochs1.average(picks=picks)
    
    # epochs2 = mne.Epochs(raw, eves, event_id=[2],  proj=True, baseline = (-0.3,0), tmin=-0.3, tmax=5.2, 
    #                      reject=dict(eeg=150e-6), picks=picks)
    # evoked2 = epochs2.average(picks=picks)
    
    # epochs3 = mne.Epochs(raw, eves, event_id=[3],  proj=True, baseline = (-0.3,0), tmin=-0.3, tmax=5.2, 
    #                      reject=dict(eeg=150e-6), picks=picks)
    # evoked3 = epochs3.average(picks=picks)

#%%Saving epochs and evoked to mat file 
    # a = (epochs.get_data(picks)).dtype=np.int64
    # b = epochs_20.get_data(picks)
    a= evoked.get_data(picks)
    # b = evoked1.get_data(picks)
    # c = evoked2.get_data(picks)
    # d = evoked3.get_data(picks)
    # # z = evoked_12.get_data(picks)
    
    t=epochs.times
    # mat_ids_ep = dict(epochs=a, epochs20=b, fs=4096, t=epochs.times)
    mat_ids_ev = dict(evoked=a, fs=4096, t=epochs.times)
    #evoked1 = b, evoked2 = c, evoked3 = c,
    # savemat(save_mat_loc + subj + '_allepochs0.1_NB.mat', mat_ids_ep)
    savemat(save_mat_loc + subj + '_GDTevokedall.mat', mat_ids_ev)
    
    print('WOOOHOOOO! Saved ' + subj)

    del  evoked