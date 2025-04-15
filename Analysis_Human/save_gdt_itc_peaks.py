# -*- coding: utf-8 -*-
"""
Created on Thu May 25 21:10:02 2023

@author: vmysorea
"""
from scipy.stats import sem
import numpy as np
from scipy import io
from matplotlib import pyplot as plt
import sys
from scipy.io import savemat
import seaborn as sns
import mne
sys.path.append('C:/Users/vmysorea/mne-python/')
sys.path.append('C:/Users/vmysorea/ANLffr/')

#%% Initializing..
data_loc = ('D:/PhD/Stim_Analysis/GapDetection_EEG/AnalyzedFiles_Figures/GDT_matfiles/')

subjlist = ['S069',  'S072', 'S078', 'S088', 'S104',
            'S105', 'S259', 'S260', 'S268', 'S269',
            'S270', 'S271', 'S273', 'S274', 'S277',
            'S279', 'S280', 'S281', 'S282', 'S284',
            'S285', 'S288', 'S290', 'S291', 'S303',
            'S308', 'S310', 'S312', 'S337',
            'S339', 'S340', 'S341', 'S342', 'S344',
            'S345', 'S347', 'S352', 'S355', 'S358']

# subjlist = ['S069',  'S072', 'S078', 'S088', 'S104',
#             'S105', 'S259', 'S260', 'S268', 'S269',
#             'S270', 'S271', 'S273', 'S274', 'S277',
#             'S279', 'S280', 'S281', 'S282', 'S284',
#             'S285', 'S288', 'S290', 'S303',
#             'S305', 'S308', 'S310', 'S312', 'S337',
#             'S339', 'S340', 'S341', 'S342', 'S344',
#             'S345', 'S347', 'S352', 'S355', 'S358']

evk16 = np.zeros((len(subjlist),3073))
evk32 = np.zeros((len(subjlist),3073))
evk64 = np.zeros((len(subjlist),3073))
itc16 = np.zeros((len(subjlist),3073))
itc32 = np.zeros((len(subjlist),3073))
itc64 = np.zeros((len(subjlist),3073))
onset16 = np.zeros((len(subjlist),10241))
onset32 = np.zeros((len(subjlist),10241))
onset64 = np.zeros((len(subjlist),10241))

chans = ['A5','A26','A31','A32']
picks = [0, 1, 2, 3]
# chans = ['A32']
# picks = [3] #If considering only A32

for subj in range(len(subjlist)):
    sub = subjlist [subj]
    dat = io.loadmat(data_loc + sub + '_evoked(4chan)_ITC_BInc.mat', squeeze_me=True)
    dat.keys()
    onset_16 = dat['onset_16'][picks]
    onset_32 = dat['onset_32'][picks]
    onset_64 = dat['onset_64'][picks]
    evoked16 = dat['evoked_1'][picks]
    evoked32 = dat['evoked_2'][picks]
    evoked64 = dat['evoked_3'][picks]
    itc1 = dat['itc1'][picks]
    itc2 = dat['itc2'][picks]
    itc3 = dat['itc3'][picks]
    fs = dat['fs']
    t_full = dat['t']    #Full plotting
    t = dat ['t1']     #GDT and ITC
    n_channels = dat['n_channels']
    freqs=dat['freqs']
    evk16[subj,:] = evoked16.mean(axis=0)*1e6   #Mean of EEG channels
    evk32[subj,:] = evoked32.mean(axis=0)*1e6
    evk64[subj,:] = evoked64.mean(axis=0)*1e6
    onset16[subj,:] = onset_16.mean(axis=0)*1e6
    onset32[subj,:] = onset_32.mean(axis=0)*1e6
    onset64[subj,:] = onset_64.mean(axis=0)*1e6
    itc16[subj,:] = itc1.mean(axis=0)*1e6
    itc32[subj,:] = itc2.mean(axis=0)*1e6
    itc64[subj,:] = itc3.mean(axis=0)*1e6

#Mean across subjects -- required only for plotting
evk16_avg = evk16.mean(axis=0)
evk16_sem = sem(evk16)
evk32_avg = evk32.mean(axis=0)
evk32_sem = sem(evk32)
evk64_avg = evk64.mean(axis=0)
evk64_sem = sem(evk64)

itc16_avg = itc16.mean(axis=0)
itc16_sem = sem(itc16)
itc32_avg = itc32.mean(axis=0)
itc32_sem = sem(itc32)
itc64_avg = itc64.mean(axis=0)
itc64_sem = sem(itc64)

onset16_avg = onset16.mean(axis=0)
onset16_sem = sem(onset16)
onset32_avg = onset32.mean(axis=0)
onset32_sem = sem(onset32)
onset64_avg = onset64.mean(axis=0)
onset64_sem = sem(onset64)
#%% Saving GDT -- P1, P2 - amp and latencies for the onsets, and gap responses

#Saving only one max peak from 0-0.3 seconds -- for gap evoked and ITC
# t1_start, t1_end = -0.1, 0.3

# gap_results = {}

# evkds_all = {'gap16':evk16,
#              'gap32':evk32,
#              'gap64':evk64,
#              'itc16':itc16,
#              'itc32':itc32,
#              'itc64':itc64}

# # Loop through conditions
# for condition, evkds in evkds_all.items():
#     peak = []
#     latency = []

#     # Iterate through each evoked dataset
#     for evkd in evkds:
#         # Find indices corresponding to the time slots
#         t1_indices = np.where((t >= t1_start) & (t <= t1_end))[0]

#         # Extract data within the time slots
#         data_t1 = evkd[t1_indices]

#         # Find the index corresponding to the peak value in each time slot
#         peak_index_t1 = t1_indices[data_t1.argmax()]

#         # Ensure the peak picked is positive, if not, find the closest positive peak!
#         if data_t1.max() < 0:
#             nearest_positive_index = t1_indices[(data_t1 > 0).argmax()] if (data_t1 > 0).any() else -1
#             if nearest_positive_index != -1:
#                 peak_value_t1 = data_t1[nearest_positive_index]
#                 peak_latency_t1 = t[nearest_positive_index]
#             else:
#                 # No positive values found, set values to None or a suitable default
#                 peak_value_t1 = np.NAN
#                 peak_latency_t1 = np.NAN
#         else:
#             # Peak value is positive
#             peak_value_t1 = data_t1.max() #Get peak amplitude
#             peak_latency_t1 = t[peak_index_t1]     #Get latency in s from the index

#         peak.append(peak_value_t1)
#         latency.append(peak_latency_t1)

#     # Store results in the dictionary
#     gap_results[condition] = {'subject':subjlist,
#                               'peak': peak,
#                               'latency': latency,
#                               'channels':chans}

### Saving P1 and P2 responses  for gap responses and ITC

tP1_start, tP1_end = -0.1, 0.1
tP2_start, tP2_end = 0.1, 0.3

gap_results_bothpeaks = {}

evkds_all_1 = {'gap16_1':evk16,
               'gap32_1':evk32,
               'gap64_1':evk64,
               'itc16_1':itc16,
               'itc32_1':itc32,
               'itc64_1':itc64}

# Loop through conditions
for condition, evkds in evkds_all_1.items():
    P1_peak = []
    P1_latency = []
    P2_peak = []
    P2_latency = []
    P1_mean = []
    P2_mean = []

    # Iterate through each evoked dataset
    for evkd in evkds:
        # Find indices corresponding to the time slots
        t1_indices = np.where((t >= tP1_start) & (t <= tP1_end))[0]
        t2_indices = np.where((t >= tP2_start) & (t <= tP2_end))[0]

        # Extract data within the time slots
        data_t1 = evkd[t1_indices]
        data_t2 = evkd[t2_indices]

        # Find the index corresponding to the peak value in each time slot
        peak_index_t1 = t1_indices[data_t1.argmax()]
        peak_index_t2 = t2_indices[data_t2.argmax()]

        # Ensure the peak picked is positive, if not, find the closest positive peak!
        if data_t1.max() < 0:
            nearest_positive_index = t1_indices[(data_t1 > 0).argmax()] if (data_t1 > 0).any() else -1
            if nearest_positive_index != -1:
                peak_value_t1 = data_t1[nearest_positive_index]
                peak_latency_t1 = t[nearest_positive_index]

            else:
                # No positive values found, set values to None or a suitable default
                peak_value_t1 = np.NAN
                peak_latency_t1 = np.NAN

        else:
            # Peak value is positive
            peak_value_t1 = data_t1.max() #Get peak amplitude
            peak_latency_t1 = t[peak_index_t1]     #Get latency in s from the index

        if data_t2.max() < 0:
           nearest_positive_index_t2 = t2_indices[(data_t2 > 0).argmax()] if (data_t2 > 0).any() else -1
           if nearest_positive_index_t2 != -1:
               peak_value_t2 = data_t2[nearest_positive_index_t2]
               peak_latency_t2 = t[nearest_positive_index_t2]
           else:
               # No positive values found, set values to None or a suitable default
               peak_value_t2 = np.NAN
               peak_latency_t2 = np.NAN

        else:
            # Peak value in P2 is positive
            peak_value_t2 = data_t2.max()
            peak_latency_t2 = t[peak_index_t2]

         # Compute mean amplitudes in P1 and P2 windows
        mean_value_t1 = np.mean(data_t1)
        mean_value_t2 = np.mean(data_t2)

       # Store
        P1_mean.append(mean_value_t1)
        P2_mean.append(mean_value_t2)

        P1_peak.append(peak_value_t1)
        P1_latency.append(peak_latency_t1)
        P2_peak.append(peak_value_t2)
        P2_latency.append(peak_latency_t2)

    # Store results in the dictionary
    gap_results_bothpeaks[condition] = {'subject':subjlist,
                                        'P1_peak': P1_peak,
                                        'P1_latency': P1_latency,
                                        'P2_peak': P2_peak,
                                        'P2_latency': P2_latency,
                                        'P1_mean': P1_mean,
                                        'P2_mean': P2_mean,
                                        'channels':chans}

##Saving evoked onset responses -- for the full duration

tP1_start, tP1_end = 0, 0.1
tP2_start, tP2_end = 0.1, 0.3

onset_results = {}

evkds_all_2 = {'onset16':onset16,
             'onset32':onset32,
             'onset64':onset64}

# Loop through conditions
for condition, evkds in evkds_all_2.items():
    P1_peak = []
    P1_latency = []
    P2_peak = []
    P2_latency = []

    # Iterate through each evoked dataset
    for evkd in evkds:
        # Find indices corresponding to the time slots
        t1_indices = np.where((t_full >= tP1_start) & (t_full <= tP1_end))[0]
        t2_indices = np.where((t_full > tP2_start) & (t_full <= tP2_end))[0]

        # Extract data within the time slots
        data_t1 = evkd[t1_indices]
        data_t2 = evkd[t2_indices]

        # Find the index corresponding to the peak value in each time slot
        peak_index_t1 = t1_indices[data_t1.argmax()]
        peak_index_t2 = t2_indices[data_t2.argmax()]

        # Ensure the peak picked is positive, if not, find the closest positive peak!
        if data_t1.max() < 0:
            nearest_positive_index = t1_indices[(data_t1 > 0).argmax()] if (data_t1 > 0).any() else -1
            if nearest_positive_index != -1:
                peak_value_t1 = data_t1[nearest_positive_index]
                peak_latency_t1 = t_full[nearest_positive_index]

            else:
                # No positive values found, set values to None or a suitable default
                peak_value_t1 = np.NAN
                peak_latency_t1 = np.NAN

        else:
        # Peak value is positive
            peak_value_t1 = data_t1.max() #Get peak amplitude
            peak_latency_t1 = t_full[peak_index_t1]     #Get latency in s from the index

        if data_t2.max() < 0:
           nearest_positive_index_t2 = t2_indices[(data_t2 > 0).argmax()] if (data_t2 > 0).any() else -1
           if nearest_positive_index_t2 != -1:
               peak_value_t2 = data_t2[nearest_positive_index_t2]
               peak_latency_t2 = t_full[nearest_positive_index_t2]

           else:
               # No positive values found, set values to None or a suitable default
               peak_value_t2 = np.NAN
               peak_latency_t2 = np.NAN

        else:
            # Peak value in P2 is positive
            peak_value_t2 = data_t2.max()
            peak_latency_t2 = t_full[peak_index_t2]

        P1_peak.append(peak_value_t1)
        P1_latency.append(peak_latency_t1)
        P2_peak.append(peak_value_t2)
        P2_latency.append(peak_latency_t2)

    # Store results in the dictionary
    onset_results[condition] = {'subject':subjlist,
                                'P1_peak': P1_peak,
                                'P1_latency': P1_latency,
                                'P2_peak': P2_peak,
                                'P2_latency': P2_latency,
                                'channels':chans}

# gap_results |
GDT_results = onset_results |  gap_results_bothpeaks #Merging the two dictionaries

savemat(data_loc + 'GDT_Evoked_ITC_4chan_-0.1-0.3_WithMeanAmp.mat',GDT_results)

###Checking if the manual peak picking is similar to the MNE peak picking -- Verified it is right
# info = mne.create_info(ch_names=['A32'], sfreq=4096, ch_types='eeg', verbose=None)
# gap_16 = mne.EvokedArray(evoked16, info, tmin=t1[0])
# gap_16.get_peak(ch_type=None, tmin=0, tmax=0.3,mode='abs', return_amplitude=True)

# gap_32 = mne.EvokedArray(evoked32, info, tmin=t1[0])
# gap_32.get_peak(ch_type=None, tmin=0, tmax=0.3,mode='abs', return_amplitude=True)

# onset_16 = mne.EvokedArray(onset16, info, tmin=t[0])
# onset_16.get_peak(ch_type=None, tmin=0.15, tmax=0.3,mode='pos', return_amplitude=True)


#%%%## Saving steady state to ensure input to SVM is the same as binding EEG input

t1 = t>=0.3
t2 = t<=0.5
t3 = np.array([t2[i] and t1[i] for i in range(len(t1))])    #Before gap

SS16 = (evk16[:,t3]).mean(axis=1)
SS32 = (evk32[:,t3]).mean(axis=1)
SS64 = (evk64[:,t3]).mean(axis=1)

plt.bar(('16','32', '64'), (SS16.mean(axis=0), SS32.mean(axis=0), SS64.mean()))
plt.show()

mat_id = dict(sub=subjlist,SS16=SS16, SS32=SS32,SS64=SS64)

savemat(data_loc + 'AllSubj_GDTSS(4chan)_1-40Hz_1sec.mat', mat_id)

#%%Plotting -- Manual
# Plot entire duration - All subjects
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

plt.savefig(data_loc + 'GDT_Evoked_4chan_AllSubj.png')

#%%##Plotting -All three gap duration responses in the same plot
# fig, ax = plt.subplots(constrained_layout=True)
# plt.errorbar(t, evk16_avg, yerr=evk16_sem,color='darkblue', linewidth=2, ecolor='lightsteelblue', label = 'Gap 16 ms')
# # plt.errorbar(t, evk32_avg, yerr=evk32_sem,color='purple', linewidth=2, ecolor='thistle', label='Gap 32 ms')
# # plt.errorbar(t, evk64_avg, yerr=evk64_sem,color='green', linewidth=2, ecolor='palegreen', label='Gap 64 ms')
# plt.xlim([-0.4, 1.1])
# # plt.ylim([0.02, 0.09])
# plt.xlabel('Time (in seconds)')
# plt.ylabel('Amplitude(\u03bcV)')
# #fig.text(-0.03,0.5, 'itc Value', va='center',rotation ='vertical')
# fig.suptitle('Evoked Responses (N=' + str(len(subjlist)) + ')', fontsize=14)
# params = {'legend.fontsize': 'xx-small',
#           'figure.figsize': (6, 5),
#           'ytick.labelsize': 'xx-small'}
#           #'ytick.major.pad': '6'}
# plt.rcParams.update(params)
# plt.tight_layout(rect=[0, 0.03, 1, 0.95])
# plt.legend()
# #fig.supylabel('itc Value')
# plt.show()

# #%% Plot full responses

# # Plot full responses across 3 subplots for each duration
# fig, ax = plt.subplots(3, 1, sharex=True, sharey=True, constrained_layout=True)
# ax[0].errorbar(t_full, onset16_avg, yerr=onset16_sem,color='darkblue', linewidth=1.5, ecolor='lightsteelblue')
# ax[0].set_title('Gap 16 ms', loc='center', fontsize=12)
# ax[1].errorbar(t_full, onset32_avg, yerr=onset32_sem,color='purple', linewidth=1.5, ecolor='thistle')
# ax[1].set_title('Gap 32 ms', loc='center', fontsize=12)
# ax[2].errorbar(t_full, onset64_avg, yerr=onset64_sem,color='green', linewidth=1.5, ecolor='palegreen')
# ax[2].set_title('Gap 64 ms', loc='center', fontsize=12)
# # plt.xlim([-0.1, 1.1])
# # plt.ylim([0.02, 0.09])
# plt.xlabel('Time (in seconds)')
# ax[1].set_ylabel('Amplitude(\u03bcV)')
# #fig.text(-0.03,0.5, 'itc Value', va='center',rotation ='vertical')
# fig.suptitle('Evoked responses for gaps (N=' + str(len(subjlist)) + ')', fontsize=14)
# params = {'legend.fontsize': 'x-large',
#           'figure.figsize': (6, 5),
#           'ytick.labelsize': 'xx-small'}
#           #'ytick.major.pad': '6'}
# plt.rcParams.update(params)
# plt.tight_layout(rect=[0, 0.03, 1, 0.95])
# #fig.supylabel('itc Value')
# plt.show()

# ###Plot full responses in the same plot for all 3 gap durations
# from matplotlib.colors import ListedColormap
# cmap = ListedColormap(sns.color_palette())

# palette = sns.color_palette("rocket_r", as_cmap=True)

# fig, ax = plt.subplots(constrained_layout=True)
# plt.errorbar(t_full, onset16_avg, yerr=onset16_sem,color='darkblue', linewidth=2, ecolor='lightsteelblue',alpha=0.5,label='Gap 16 ms')
# # plt.errorbar(t_full, onset32_avg, yerr=onset32_sem,color='purple', linewidth=2, ecolor='thistle',alpha=0.5, label='Gap 32 ms')
# plt.errorbar(t_full, onset64_avg, yerr=onset64_sem,color='green', linewidth=2, ecolor='palegreen',alpha=0.5, label='Gap 64 ms')
# # plt.xlim([-0.2, 0.5])
# # plt.ylim([0.02, 0.09])
# plt.xlabel('Time (in seconds)')
# plt.ylabel('Amplitude(\u03bcV)')
# #fig.text(-0.03,0.5, 'itc Value', va='center',rotation ='vertical')
# fig.suptitle('Evoked Responses (N=' + str(len(subjlist)) + ')', fontsize=14)
# params = {'legend.fontsize': 'xx-small',
#           'figure.figsize': (6, 5),
#           'ytick.labelsize': 'xx-small'}
#           #'ytick.major.pad': '6'}
# plt.rcParams.update(params)
# plt.tight_layout(rect=[0, 0.03, 1, 0.95])
# plt.legend()
# #fig.supylabel('itc Value')
# plt.show()

# #%% Plot ITC

# fig, ax = plt.subplots(3, 1, sharex=True, sharey=True, constrained_layout=True)
# ax[0].errorbar(t, itc16_avg, yerr=itc16_sem,color='darkblue', linewidth=1.5, ecolor='lightsteelblue')
# ax[0].set_title('Gap 16 ms', loc='center', fontsize=12)
# ax[1].errorbar(t, itc32_avg, yerr=itc32_sem,color='purple', linewidth=1.5, ecolor='thistle')
# ax[1].set_title('Gap 32 ms', loc='center', fontsize=12)
# ax[2].errorbar(t, itc64_avg, yerr=itc64_sem,color='green', linewidth=1.5, ecolor='palegreen')
# ax[2].set_title('Gap 64 ms', loc='center', fontsize=12)
# # plt.xlim([-0.1, 1.1])
# # plt.ylim([0.02, 0.09])
# plt.xlabel('Time (in seconds)')
# ax[1].set_ylabel('Amplitude(\u03bcV)')
# #fig.text(-0.03,0.5, 'itc Value', va='center',rotation ='vertical')
# fig.suptitle('Evoked Responses (N=' + str(len(subjlist)) + ')', fontsize=14)
# params = {'legend.fontsize': 'xx-small',
#           'figure.figsize': (6, 5),
#           'ytick.labelsize': 'xx-small'}
#           #'ytick.major.pad': '6'}
# plt.rcParams.update(params)
# plt.tight_layout(rect=[0, 0.03, 1, 0.95])
# #fig.supylabel('itc Value')
# plt.show()

