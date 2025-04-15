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
<<<<<<< HEAD
from scipy.stats import sem
=======
>>>>>>> refs/remotes/origin/main

plt.switch_backend('QT5Agg')  # Making the plots interactive (Scrollable)
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')
# Defining the dimensions and quality of figures
plt.rcParams['figure.dpi'] = 120
plt.rcParams["figure.figsize"] = (5.5, 5)
<<<<<<< HEAD
=======
plt.tight_layout
>>>>>>> refs/remotes/origin/main

# %% Loading subjects, reading data, mark bad channels
froot = 'D:/PhD/Data/MTB_EP - GDT, Binding, mTRF/GDT/'  # file location
save_loc = ('D:/PhD/Stim_Analysis/GapDetection_EEG/AnalyzedFiles_Figures/GDT_matfiles/')

<<<<<<< HEAD
### S305 not done -- saved on stim channel (ignore this subject)
subjlist = ['S291']

# subjlist = ['S069',  'S072', 'S078', 'S088', 'S104',
#             'S105', 'S259', 'S260', 'S268', 'S269',
#             'S270', 'S271', 'S273', 'S274', 'S277',
#             'S279', 'S280', 'S281', 'S282', 'S284',
#             'S285', 'S288', 'S290', 'S303',
#             'S305', 'S308', 'S310', 'S312', 'S337',
#             'S339', 'S340', 'S341', 'S342', 'S344',
#             'S345', 'S347', 'S352', 'S355', 'S358']

 # Load subject folder
=======
subjlist =  ['S305'] # Load subject folder
>>>>>>> refs/remotes/origin/main
condlist = [1, 2, 3]  # List of conditions- Here 3 GDs - 16, 32, 64 ms
condnames = ['0.016 s', '0.032 s', '0.064 s']

for subj in subjlist:
    evokeds = []
<<<<<<< HEAD

    # Load data and read event channel
    fpath = froot + subj + '/'
    bdfs = fnmatch.filter(os.listdir(fpath), subj +'_GDT*.bdf')

    print('LOADING! ' + subj +' raw data')

=======
    
    # Load data and read event channel
    fpath = froot + subj + '/'
    bdfs = fnmatch.filter(os.listdir(fpath), subj +'_GDT*.bdf')
    
    print('LOADING! ' + subj +' raw data')
    
>>>>>>> refs/remotes/origin/main
    # Load data and read event channel
    rawlist = []
    evelist = []

    for k, rawname in enumerate(bdfs):
        rawtemp, evestemp = bs.importbdf(fpath + rawname, verbose='DEBUG',refchans=['EXG1', 'EXG2'])
<<<<<<< HEAD

        if rawtemp.info['sfreq'] != 4096:
            rawtemp = rawtemp.resample(4096, npad="auto")

        rawlist += [rawtemp, ]
        evelist += [evestemp, ]

    raw, eves = mne.concatenate_raws(rawlist, events_list=evelist)
=======
        
        if rawtemp.info['sfreq'] != 4096:
            rawtemp = rawtemp.resample(4096, npad="auto")
            
        rawlist += [rawtemp, ]
        evelist += [evestemp, ]
            
    raw, eves = mne.concatenate_raws(rawlist, events_list=evelist)
    
>>>>>>> refs/remotes/origin/main

    # To check and mark bad channels
    # raw.plot(duration=25.0, n_channels=32, scalings=dict(eeg=150e-6))

#%% Bad channels for each subject
<<<<<<< HEAD

    if subj == ['S273','S285']:
       raw.info['bads'].append('A24')
       raw.info['bads'].append('A7')

    if subj == 'S268':
       raw.info['bads'].append('A25')
       raw.info['bads'].append('A29')

    if subj == 'S269':
       raw.info['bads'].append('A16')
       raw.info['bads'].append('A7')

=======
  
    if subj == ['S273','S285']:
       raw.info['bads'].append('A24')
       raw.info['bads'].append('A7')
      
    if subj == 'S268':
       raw.info['bads'].append('A25')
       raw.info['bads'].append('A29')
    
    if subj == 'S269':
       raw.info['bads'].append('A16')
       raw.info['bads'].append('A7')
       
>>>>>>> refs/remotes/origin/main
    if subj == 'S274':
       raw.info['bads'].append('A23')
       raw.info['bads'].append('A28')
       raw.info['bads'].append('A19')
       raw.info['bads'].append('A3')
<<<<<<< HEAD

=======
       
>>>>>>> refs/remotes/origin/main
    if subj == 'S282':
       raw.info['bads'].append('A24')
       raw.info['bads'].append('A25')
       raw.info['bads'].append('A7')
<<<<<<< HEAD

=======
       
>>>>>>> refs/remotes/origin/main
    if subj == 'S277':
        raw.info['bads'].append('A1')
        raw.info['bads'].append('A17')
        raw.info['bads'].append('A30')
<<<<<<< HEAD

=======
    
>>>>>>> refs/remotes/origin/main
    if subj == 'S279':
        raw.info['bads'].append('A23')
        raw.info['bads'].append('A24')
        raw.info['bads'].append('A28')
<<<<<<< HEAD

=======
        
>>>>>>> refs/remotes/origin/main
    if subj == 'S280':
        raw.info['bads'].append('A3')
        raw.info['bads'].append('A24')
        raw.info['bads'].append('A25')
        raw.info['bads'].append('A6')
        raw.info['bads'].append('A7')
        raw.info['bads'].append('A28')
        raw.info['bads'].append('A30')
        raw.info['bads'].append('A1')
<<<<<<< HEAD

=======
        
>>>>>>> refs/remotes/origin/main
    if subj == 'S259':
        raw.info['bads'].append('A3')
        raw.info['bads'].append('A24')
        raw.info['bads'].append('A25')
        raw.info['bads'].append('A6')
        raw.info['bads'].append('A7')
        raw.info['bads'].append('A26')
        raw.info['bads'].append('A21')
<<<<<<< HEAD

    if subj == 'S270':
        raw.info['bads'].append('A25')
        raw.info['bads'].append('A3')

=======
        
    if subj == 'S270':
        raw.info['bads'].append('A25')
        raw.info['bads'].append('A3')
        
>>>>>>> refs/remotes/origin/main
    if subj == 'S271':
        raw.info['bads'].append('A7')
        raw.info['bads'].append('A3')
        raw.info['bads'].append('A28')
        raw.info['bads'].append('A6')
        raw.info['bads'].append('A25')
        raw.info['bads'].append('A1')
<<<<<<< HEAD

=======
    
>>>>>>> refs/remotes/origin/main
    if subj == 'S281':
        raw.info['bads'].append('A7')
        raw.info['bads'].append('A24')
        raw.info['bads'].append('A28')
        raw.info['bads'].append('A21')
        raw.info['bads'].append('A25')
        raw.info['bads'].append('A15')
<<<<<<< HEAD

=======
        
>>>>>>> refs/remotes/origin/main
    if subj == 'S290':
        raw.info['bads'].append('A7')
        raw.info['bads'].append('A3')
        raw.info['bads'].append('A24')
        raw.info['bads'].append('A30')
        raw.info['bads'].append('A1')
        raw.info['bads'].append('A20')
        raw.info['bads'].append('A28')
<<<<<<< HEAD

=======
      
>>>>>>> refs/remotes/origin/main
    if subj == 'S288':
        raw.info['bads'].append('A27')
        raw.info['bads'].append('A28')
        raw.info['bads'].append('A30')
        raw.info['bads'].append('A1')
<<<<<<< HEAD


    if subj == 'S284':
       raw.info['bads'].append('A25')
       raw.info['bads'].append('A28')

=======
      
       
    if subj == 'S284':
       raw.info['bads'].append('A25')
       raw.info['bads'].append('A28')
       
>>>>>>> refs/remotes/origin/main
    if subj == 'S305':
       raw.info['bads'].append('A7')
       raw.info['bads'].append('A25')
       raw.info['bads'].append('A28')
<<<<<<< HEAD

=======
       
>>>>>>> refs/remotes/origin/main
    if subj == 'S303':
       raw.info['bads'].append('A7')
       raw.info['bads'].append('A18')
       raw.info['bads'].append('A24')
<<<<<<< HEAD

=======
       
>>>>>>> refs/remotes/origin/main
    if subj == 'S288':
       raw.info['bads'].append('A7')
       raw.info['bads'].append('A4')
       raw.info['bads'].append('A24')
<<<<<<< HEAD

=======
       
>>>>>>> refs/remotes/origin/main
    if subj == 'S260':
       raw.info['bads'].append('A9')
       raw.info['bads'].append('A10')
       raw.info['bads'].append('A11')
<<<<<<< HEAD

=======
       
>>>>>>> refs/remotes/origin/main
    if subj == 'S341':
       raw.info['bads'].append('A9')
       raw.info['bads'].append('A10')
       raw.info['bads'].append('A11')
<<<<<<< HEAD

    if subj == 'S352':
        raw.info['bads'].append('A20')
        raw.info['bads'].append('A1')

    if subj == 'S312':
        raw.info['bads'].append('A24')

    if subj == 'S347':
        raw.info['bads'].append('A3')
        raw.info['bads'].append('A25')

    if subj == 'S347':
         raw.info['bads'].append('A3')
         raw.info['bads'].append('A25')

=======
       
    if subj == 'S352':
        raw.info['bads'].append('A20')
        raw.info['bads'].append('A1')
        
    if subj == 'S312':
        raw.info['bads'].append('A24')
    
    if subj == 'S347':
        raw.info['bads'].append('A3')
        raw.info['bads'].append('A25')
        
    if subj == 'S347':
         raw.info['bads'].append('A3')
         raw.info['bads'].append('A25')
         
>>>>>>> refs/remotes/origin/main
    if subj == 'S104':
         raw.info['bads'].append('A13')
         raw.info['bads'].append('A17')
         raw.info['bads'].append('A18')
<<<<<<< HEAD
         raw.info['bads'].append('A25')



=======
    
     
        
>>>>>>> refs/remotes/origin/main
# %% Filtering
    raw.filter(1., 40.)
    raw.info

# %% Blink Rejection
    blinks = find_blinks(raw)
    # raw.plot(events=blinks, duration=25.0, n_channels=32, scalings=dict(eeg=150e-6))
    epochs_blinks = mne.Epochs(raw, blinks, event_id=998, baseline=(-0.25, 0.25),reject=dict(eeg=500e-6), tmin=-0.25, tmax=0.25)
    blink_proj = compute_proj_epochs(epochs_blinks, n_eeg=1)
    raw.add_proj(blink_proj)  # Adding the n=blink proj to the data -- removal of blink
<<<<<<< HEAD
    # raw.plot_projs_topomap()  # Visualizing the spatial filter

# %% Plotting Onset responses -- Remember, the trigs 1,2,3 correspond to the beginning of a trial containing 3 gaps of 0.016, 0.032, 0.064 ms respectively

#Plotting onsets for all events combined
    epochs = mne.Epochs(raw, eves, event_id=[1], baseline=(-0.3, 0), proj=True, tmin=-0.3, tmax=2.2, reject=dict(eeg=200e-6))
    epochs.apply_proj()
    evoked = epochs.average()
    Onset_Total = evoked.plot(titles= subj + 'Combined Onset - All 3 gaps')

#Plotting onsets for each event separately
    picks = ['A5','A26','A31','A32']
    # picks = ['A32']
    epochs_16 = mne.Epochs(raw, eves, event_id=[1], baseline=(-0.3, 0), proj=True,tmin=-0.3, tmax=2.2, reject=dict(eeg=150e-6), picks = picks)
    epochs_32 = mne.Epochs(raw, eves, event_id=[2], baseline=(-0.3, 0), proj=True,tmin=-0.3, tmax=2.2, reject=dict(eeg=150e-6), picks=picks)
    epochs_64 = mne.Epochs(raw, eves, event_id=[3], baseline=(-0.3, 0), proj=True,tmin=-0.3, tmax=2.2, reject=dict(eeg=150e-6), picks=picks)

    onset_16 = epochs_16.average(picks = picks)
    onset_32 = epochs_32.average(picks = picks)
    onset_64 = epochs_64.average(picks = picks)

    #Get P1, P2 peaks for the onset
    # onsetP1_16 = onset_16.get_peak(ch_type='eeg', tmin=0.0, tmax=0.15,mode='pos', return_amplitude=True)
    # onsetP1_32 = onset_32.get_peak(ch_type='eeg', tmin=0.0, tmax=0.15,mode='pos', return_amplitude=True)
    # onsetP1_64 = onset_64.get_peak(ch_type='eeg', tmin=0.0, tmax=0.15,mode='pos', return_amplitude=True)

=======
    # raw.plot_projs_topomap()  # Visualizing the spatial filter    

# %% Plotting Onset responses -- Remember, the trigs 1,2,3 correspond to the beginning of a trial containing 3 gaps of 0.016, 0.032, 0.064 ms respectively
  
    picks = ['A5','A26','A31','A32']
    epochs_16 = mne.Epochs(raw, eves, event_id=[1], baseline=(-0.3, 0), proj=True,tmin=-0.3, tmax=2.2, reject=dict(eeg=150e-6), picks=picks)
    epochs_32 = mne.Epochs(raw, eves, event_id=[2], baseline=(-0.3, 0), proj=True,tmin=-0.3, tmax=2.2, reject=dict(eeg=150e-6), picks=picks)
    epochs_64 = mne.Epochs(raw, eves, event_id=[3], baseline=(-0.3, 0), proj=True,tmin=-0.3, tmax=2.2, reject=dict(eeg=150e-6), picks=picks)
    
    onset_16 = epochs_16.average(picks = picks)
    onset_32 = epochs_32.average(picks = picks)
    onset_64 = epochs_64.average(picks = picks)
    
    #Get P1, P2 peaks for the onset     
    # onsetP1_16 = onset_16.get_peak(ch_type='eeg', tmin=0.0, tmax=0.15,mode='pos', return_amplitude=True)
    # onsetP1_32 = onset_32.get_peak(ch_type='eeg', tmin=0.0, tmax=0.15,mode='pos', return_amplitude=True)
    # onsetP1_64 = onset_64.get_peak(ch_type='eeg', tmin=0.0, tmax=0.15,mode='pos', return_amplitude=True)
    
>>>>>>> refs/remotes/origin/main
    # onsetP2_16 = onset_16.get_peak(ch_type='eeg', tmin=0.15, tmax=0.3,mode='pos', return_amplitude=True)
    # onsetP2_32 = onset_32.get_peak(ch_type='eeg', tmin=0.15, tmax=0.3,mode='pos', return_amplitude=True)
    # onsetP2_64 = onset_64.get_peak(ch_type='eeg', tmin=0.15, tmax=0.3,mode='pos', return_amplitude=True)

<<<<<<< HEAD
    onset_16.plot(titles='Onset - ' + subj, picks = picks)
    onset_32.plot(picks=picks, titles='Onset - ' + subj)
    onset_64.plot(picks=picks, titles='Onset - ' + subj)

# %% Creating manual events to add the responses of each gap size
    #Triggers set after each gap, remember!!!

=======
    # OnsetResponse_All3 = onset_16.plot(picks=picks, titles='Onset - ' + subj)
    # OnsetResponse_All3 = onset_32.plot(picks=picks, titles='Onset - ' + subj)
    # OnsetResponse_All3 = onset_64.plot(picks=picks, titles='Onset - ' + subj)
    
    # OnsetResponse_All3.savefig(save_loc + 'OnsetResponse_All3_.png' + subj, dpi=300)
    
    # mat_ids1 = dict(onsetP1_16=onsetP1_16, onsetP1_32=onsetP1_32, onsetP1_64=onsetP1_64,
    #                 onsetP2_16=onsetP2_16, onsetP2_32=onsetP2_32, onsetP2_64=onsetP2_64)

# %% Creating manual events to add the responses of each gap size
    #Triggers set after each gap, remember!!!
    
>>>>>>> refs/remotes/origin/main
    fs = raw.info['sfreq']
    gap_durs = [.016, .032, 0.064]
    eves_manual = np.zeros((3*eves.shape[0], 3))
    for k in range(1, eves.shape[0]):
        for m in range(3):
            current_eves = eves[k, :].copy()
            gap_samps = gap_durs[int(current_eves[2]-1)]*fs
<<<<<<< HEAD
            current_eves[0] = current_eves[0] + (((m+1) * np.round(0.5*fs)) + ((m+1)*gap_samps))  #0.5 is the tone duration | Trigger at the end of gap
            # current_eves[0] = current_eves[0] + (((m+1) * np.round(0.5*fs)) + ((m)*gap_samps))  #Trigger at the onset of gap
            eves_manual[((k-1)*3) + m, :] = current_eves

    eves_manual = np.int64(eves_manual)

    # Epoching

    epochs_1 = mne.Epochs(raw, eves_manual[:], event_id=[1], baseline=(-0.2,-0.07), proj=True,tmin=-0.2, tmax=0.55, reject=dict(eeg=150e-6), picks = picks)
    epochs_2 = mne.Epochs(raw, eves_manual[:], event_id=[2], baseline=(-0.2,-0.07), proj=True,tmin=-0.2, tmax=0.55, reject=dict(eeg=150e-6), picks=picks)
    epochs_3 = mne.Epochs(raw, eves_manual, event_id=[3], baseline=(-0.2,-0.07), proj=True,tmin=-0.2, tmax=0.55, reject=dict(eeg=150e-6), picks=picks)

    ###Baseline was (-0.1,0) -- Changed it for better responses possibly
# Averaging

    evoked_1 = epochs_1.average(picks = picks)
    evoked_2 = epochs_2.average(picks = picks)
    evoked_3 = epochs_3.average(picks = picks)

    ## Get P1, P2 peaks for the gaps
    # gapP1_16 = evoked_1.get_peak(ch_type='eeg', tmin=0.0, tmax=0.15,mode='pos', return_amplitude=True)
    # gapP1_32 = evoked_2.get_peak(ch_type='eeg', tmin=0.0, tmax=0.15,mode='pos', return_amplitude=True)
    # gapP1_64 = evoked_3.get_peak(ch_type='eeg', tmin=0.0, tmax=0.15,mode='pos', return_amplitude=True)

    # gapP2_16 = evoked_1.get_peak(ch_type='eeg', tmin=0.15, tmax=0.3,mode='pos', return_amplitude=True)
    # gapP2_32 = evoked_2.get_peak(ch_type='eeg', tmin=0.15, tmax=0.3,mode='pos', return_amplitude=True)
    # gapP2_64 = evoked_3.get_peak(ch_type='eeg', tmin=0.15, tmax=0.3,mode='pos', return_amplitude=True)

    # mat_ids2 = dict(gapP1_16=gapP1_16, gapP1_32=gapP1_32, gapP1_64=gapP1_64,
    #                gapP2_16=gapP2_16, gapP2_32=gapP2_32, gapP2_64=gapP2_64)

    # gap_ids = mat_ids2|mat_ids1
    # savemat(save_loc + subj + '_peaksA32.mat', gap_ids)


    #Plotting
    # evoked_1.plot(titles='GDT_16ms - ' + subj)
    # evoked_2.plot(ylim=dict(eeg=[-1.1, 2.5]), titles='GDT_32ms - ' + subj)
    # evoked_3.plot(ylim=dict(eeg=[-1.1, 2.5]), titles='GDT_64ms - ' + subj)

    # evokeds = dict(GDT_16ms=evoked_1, GDT_32ms=evoked_2, GDT_64ms=evoked_3)
    # mne.viz.plot_compare_evokeds(evokeds, combine='mean', title='GDT - ' +subj, picks='A32')

    ### Manual plotting
    t=epochs_1.times
    picks = [0,1,2,3]
    evk16 = evoked_1.get_data(picks)
    evk16_avg = evk16.mean(axis=0)*1e6
    evk16_sem = sem(evk16)*1e6

    evk32 = evoked_2.get_data(picks)
    evk32_avg = evk32.mean(axis=0)*1e6
    evk32_sem = sem(evk32)*1e6

    evk64= evoked_3.get_data(picks)
    evk64_avg = evk64.mean(axis=0)*1e6
    evk64_sem = sem(evk64)*1e6


    fig, ax = plt.subplots(3, 1, sharex=True, sharey=True, constrained_layout=True)
    ax[0].errorbar(t, evk16_avg, yerr=evk16_sem,color='darkblue', linewidth=1.5, ecolor='lightsteelblue')
    ax[0].set_title('Gap 16 ms', loc='center', fontsize=12)
    ax[1].errorbar(t, evk32_avg, yerr=evk32_sem,color='purple', linewidth=1.5, ecolor='thistle')
    ax[1].set_title('Gap 32 ms', loc='center', fontsize=12)
    ax[2].errorbar(t, evk64_avg, yerr=evk64_sem,color='green', linewidth=1.5, ecolor='palegreen')
    ax[2].set_title('Gap 64 ms', loc='center', fontsize=12)
    # plt.xlim([-0.1, 1.1])
    # plt.ylim([0.02, 0.09])
    plt.xlabel('Time (in seconds)')
    ax[1].set_ylabel('Amplitude(\u03bcV)')
    #fig.text(-0.03,0.5, 'itc Value', va='center',rotation ='vertical')
    fig.suptitle('Evoked Responses (N=' + str(len(subjlist)) + ')', fontsize=14)
    params = {'legend.fontsize': 'xx-small',
              'figure.figsize': (6, 5),
              'ytick.labelsize': 'xx-small'}
              #'ytick.major.pad': '6'}
    plt.rcParams.update(params)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    #fig.supylabel('itc Value')
    plt.show()

# %% Compute evoked response using ITC

# Save location for all analyzed ITC files and figs across ages
    # ITC1 - 16 ms
    freqs = np.arange(1., 14., 1.)
    n_cycles = freqs * 0.2

    power_1, itc_1 = tfr_multitaper(epochs_1, freqs, n_cycles, picks=picks,
                                    time_bandwidth=4.0, n_jobs=-1, return_itc=True)

    # # ITC2 - 32 ms
    power_2, itc_2 = tfr_multitaper(epochs_2, freqs, n_cycles, picks=picks,
                                    time_bandwidth=4.0, n_jobs=-1, return_itc=True)

    # # ITC3 - 64 ms
    power_3, itc_3 = tfr_multitaper(epochs_3, freqs, n_cycles, picks=picks,
                                    time_bandwidth=4.0, n_jobs=-1, return_itc=True)

    # Plotting ITC and Power plots for the three conditions
=======
            current_eves[0] = current_eves[0] + (((m+1) * np.round(0.5*fs)) + ((m+1)*gap_samps))  #0.5 is the tone duration
            eves_manual[((k-1)*3) + m, :] = current_eves
    
    eves_manual = np.int64(eves_manual)
    
    # Epoching
    
    epochs_1 = mne.Epochs(raw, eves_manual[:], event_id=[1], baseline=(-0.2,-0.07), proj=True,tmin=-0.2, tmax=0.55, reject=dict(eeg=150e-6), picks=picks)
    epochs_2 = mne.Epochs(raw, eves_manual, event_id=[2], baseline=(-0.2,-0.07), proj=True,tmin=-0.2, tmax=0.55, reject=dict(eeg=150e-6), picks=picks)
    epochs_3 = mne.Epochs(raw, eves_manual, event_id=[3], baseline=(-0.2,-0.07), proj=True,tmin=-0.2, tmax=0.55, reject=dict(eeg=150e-6), picks=picks)

    ###Baseline was (-0.1,0) -- Changed it for better responses possibly 
# Averaging
    
    evoked_1 = epochs_1.average(picks = picks)
    evoked_2 = epochs_2.average(picks = picks)
    evoked_3 = epochs_3.average(picks = picks)
    
    ## Get P1, P2 peaks for the gaps 
    # gapP1_16 = evoked_1.get_peak(ch_type='eeg', tmin=0.0, tmax=0.15,mode='pos', return_amplitude=True)
    # gapP1_32 = evoked_2.get_peak(ch_type='eeg', tmin=0.0, tmax=0.15,mode='pos', return_amplitude=True)
    # gapP1_64 = evoked_3.get_peak(ch_type='eeg', tmin=0.0, tmax=0.15,mode='pos', return_amplitude=True)
    
    # gapP2_16 = evoked_1.get_peak(ch_type='eeg', tmin=0.15, tmax=0.3,mode='pos', return_amplitude=True)
    # gapP2_32 = evoked_2.get_peak(ch_type='eeg', tmin=0.15, tmax=0.3,mode='pos', return_amplitude=True)
    # gapP2_64 = evoked_3.get_peak(ch_type='eeg', tmin=0.15, tmax=0.3,mode='pos', return_amplitude=True)
    
    # mat_ids2 = dict(gapP1_16=gapP1_16, gapP1_32=gapP1_32, gapP1_64=gapP1_64,
    #                gapP2_16=gapP2_16, gapP2_32=gapP2_32, gapP2_64=gapP2_64)
    
    # gap_ids = mat_ids2|mat_ids1
    # savemat(save_loc + subj + '_peaksA32.mat', gap_ids)   
    
    
    #Plotting 
    # evoked_1.plot(ylim=dict(eeg=[-1.1, 2.5]), titles='GDT_16ms - ' + subj)
    # evoked_2.plot(ylim=dict(eeg=[-1.1, 2.5]), titles='GDT_32ms - ' + subj)
    # evoked_3.plot(ylim=dict(eeg=[-1.1, 2.5]), titles='GDT_64ms - ' + subj)
    
    # evokeds = dict(GDT_16ms=evoked_1, GDT_32ms=evoked_2, GDT_64ms=evoked_3)
    # mne.viz.plot_compare_evokeds(evokeds, combine='mean', title='GDT - ' +subj, picks='A32')
    
# %% Compute evoked response using ITC

# Save location for all analyzed ITC files and figs across ages    
    # ITC1 - 16 ms
    freqs = np.arange(1., 14., 1.)
    n_cycles = freqs * 0.2
    
    power_1, itc_1 = tfr_multitaper(epochs_1, freqs, n_cycles, picks=picks,
                                    time_bandwidth=4.0, n_jobs=-1, return_itc=True)
    
    # # ITC2 - 32 ms
    power_2, itc_2 = tfr_multitaper(epochs_2, freqs, n_cycles, picks=picks,
                                    time_bandwidth=4.0, n_jobs=-1, return_itc=True)
    
    # # ITC3 - 64 ms
    power_3, itc_3 = tfr_multitaper(epochs_3, freqs, n_cycles, picks=picks,
                                    time_bandwidth=4.0, n_jobs=-1, return_itc=True)
    
    # Plotting ITC and Power plots for the three conditions    
>>>>>>> refs/remotes/origin/main
    # Cond 1
    # power_1.plot([0], baseline=(-0.3, 0), mode='mean',title='Gap duration of 16 ms - Power')
    # itc_1.plot(title='Gap duration of 16 ms - Intertrial Coherence (' + subj + ')',  baseline=(-0.1,0), combine='mean')
    # # #plt.savefig(save_loc + 'ITC1_S105.png', dpi=300)
<<<<<<< HEAD

=======
    
>>>>>>> refs/remotes/origin/main
    # # # Cond 2
    # # power_2.plot([0], baseline=(-0.3, 0), mode='mean',title='Gap duration of 32 ms - Power')
    # itc_2.plot(title='Gap duration of 32 ms- Intertrial Coherence (' + subj + ')',  baseline=(-0.1,0), combine='mean')
    # # #plt.savefig(save_loc + 'ITC1_S105.png', dpi=300)
<<<<<<< HEAD

=======
    
>>>>>>> refs/remotes/origin/main
    # # # Cond 3
    # # power_3.plot([0], baseline=(-0.3, 0), mode='mean', title='Gap duration of 64 ms - Power')
    # itc_3.plot(title='Gap duration of 64 ms - Intertrial Coherence  (' + subj + ')',baseline=(-0.1,0), combine='mean')
    # #plt.savefig(save_loc + 'ITC1_S105.png', dpi=300)
<<<<<<< HEAD

    # Saving ITC measures into mat file -- Taking the mean across the third row
    #Saving evokeds, just in case I need it for overall plotting -- Might make a difference

=======
    
    # Saving ITC measures into mat file -- Taking the mean across the third row
    #Saving evokeds, just in case I need it for overall plotting -- Might make a difference 
        
>>>>>>> refs/remotes/origin/main
    a = onset_16.get_data(picks)
    b = onset_32.get_data(picks)
    c = onset_64.get_data(picks)
    d = evoked_1.get_data(picks)
    e = evoked_2.get_data(picks)
    f = evoked_3.get_data(picks)
    x = (itc_1.data).mean(axis=1)    #Mean of the 14 freqs
    y = (itc_2.data).mean(axis=1)
    z = (itc_3.data).mean(axis=1)
<<<<<<< HEAD

=======
    
>>>>>>> refs/remotes/origin/main
    t=epochs_16.times
    t1=epochs_1.times
    mat_ids = dict(onset_16=a,onset_32=b,onset_64=c,evoked_1=d,evoked_2=e,evoked_3=f, fs=4096, t=t, t1=t1,
                    itc1=x, itc2=y, itc3=z, freqs=freqs, n_channels=picks, n_cycles=n_cycles)
<<<<<<< HEAD

    savemat(save_loc + subj + '_evoked(4chan)_ITC_BInc.mat', mat_ids)


    print('WOOOHOOOO! Saved ' + str(subj) + '!!')

=======
    
    savemat(save_loc + subj + '_evoked(4chan)_ITC_BInc.mat', mat_ids)
    
    
    print('WOOOHOOOO! Saved ' + subj)
        
>>>>>>> refs/remotes/origin/main
    # del epochs_16,epochs_32,epochs_64, epochs_1, epochs_2,epochs_3,onset_16, onset_32,onset_64,
    # evoked_1,evoked_2,evoked_3, eves_manual, eves, itc_1,itc_2,itc_3