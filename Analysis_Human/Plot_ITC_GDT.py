# -*- coding: utf-8 -*-
"""
Created on Thu Jan 27 01:59:03 2022

@author: vmysorea
"""
from scipy.stats import sem
import numpy as np
from scipy import io
from matplotlib import pyplot as plt
import sys
from scipy.io import savemat
sys.path.append('C:/Users/vmysorea/mne-python/')
sys.path.append('C:/Users/vmysorea/ANLffr/')

# %% Plotting ITC for all subjects

# Loading data to plot ITC of ALL subjects - NOT ACROSS AGE

data_loc = 'C:/Users/vmysorea/Desktop/PhD/Stim_Analysis/GapDetection_EEG/Analysis/AnalyzedFiles_Figures/All_Ages/ITC_matfiles/'
Subjects = ['S273','S268','S269']

save_loc = 'C:/Users/vmysorea/Desktop/PhD/Stim_Analysis/GapDetection_EEG/Analysis/AnalyzedFiles_Figures/All_Ages/ITC_Figures/'

# ITC 1

ITC1_Mean = np.zeros((len(Subjects), 5736))
ITC1_300ms_avg = np.zeros((len(Subjects), 1229))

for sub in range(len(Subjects)):
    subject = Subjects[sub]
    dat = io.loadmat(data_loc + 'ITC1_' + subject, squeeze_me=True)

    dat.keys()
    itc1 = dat['itc1']
    freqs = dat['freqs']
    # n_channels=dat['n_channels']
    t1 = dat['t']

    itc1_avg = itc1[:, :].mean(axis=0)
    ITC1_Mean[sub, :] = itc1_avg

    ti = t1>=0
    tj = t1<=0.3
    tnew1 = np.array([ti[i] and tj[i] for i in range(len(ti))])
    ITC1_300ms = itc1[:,tnew1].mean(axis=0)
    ITC1_300ms_avg[sub, :] = ITC1_300ms

ITC1_Mean_total = ITC1_Mean.mean(axis=0)
#ITC1_std = np.std(ITC1_Mean, axis=0)
ITC1_sem = sem(ITC1_Mean)
#plt.errorbar(t1, ITC1_Mean_total, yerr=ITC1_std, color='red')
#figure1 = plt.plot(t1,ITC1_Mean_total)
# plt.xlim([-0.1,1.1])

ITC1_300ms_avg_total = ITC1_300ms_avg.mean(axis=1)

ITC1_Peak = ITC1_Mean_total[t1 > 0]
print(max(ITC1_Peak))
#print((ITC1_300ms))

# ITC 2

ITC2_Mean = np.zeros((len(Subjects), 5736))
ITC2_300ms_avg = np.zeros((len(Subjects), 1229))

for sub in range(len(Subjects)):
    subject = Subjects[sub]
    dat = io.loadmat(data_loc + 'ITC2_' + subject, squeeze_me=True)

    dat.keys()
    itc2 = dat['itc2']
    freqs = dat['freqs']
    # n_channels=dat['n_channels']
    t2 = dat['t']

    itc2_avg = itc2[:, :].mean(axis=0)
    ITC2_Mean[sub, :] = itc2_avg

    ti = t2>=0
    tj = t2<=0.3
    tnew2 = np.array([ti[i] and tj[i] for i in range(len(ti))])
    ITC2_300ms = itc2[:,tnew2].mean(axis=0)
    ITC2_300ms_avg[sub, :] = ITC2_300ms

ITC2_Mean_total = ITC2_Mean.mean(axis=0)
#ITC2_std = np.std(ITC2_Mean, axis=0)
ITC2_sem = sem(ITC2_Mean)
#plt.errorbar(t2, ITC2_Mean_total, yerr=ITC2_sem, color='red')
#figure2 = plt.plot(t2,ITC2_Mean_total)
# plt.xlim([-0.1,1.1])

ITC2_300ms_avg_total = ITC2_300ms_avg.mean(axis=1)

ITC2_Peak = ITC2_Mean_total[t2 > 0]
print(max(ITC2_Peak))
#print(ITC2_300ms)

# ITC 3
ITC3_Mean = np.zeros((len(Subjects), 5736))
ITC3_300ms_avg = np.zeros((len(Subjects), 1229))

for sub in range(len(Subjects)):
    subject = Subjects[sub]
    dat = io.loadmat(data_loc + 'ITC3_' + subject, squeeze_me=True)

    dat.keys()
    itc3 = dat['itc3']
    freqs = dat['freqs']
    # n_channels=dat['n_channels']
    t3 = dat['t']

    itc3_avg = itc3[:, :].mean(axis=0)
    ITC3_Mean[sub, :] = itc3_avg

    ti = t3>=0
    tj = t3<=0.3
    tnew3 = np.array([ti[i] and tj[i] for i in range(len(ti))])
    ITC3_300ms = itc3[:,tnew3].mean(axis=0)
    ITC3_300ms_avg[sub, :] = ITC3_300ms

ITC3_Mean_total = ITC3_Mean.mean(axis=0)
#ITC3_std = np.std(ITC3_Mean, axis=0)
ITC3_sem = sem(ITC3_Mean)
#print(ITC3_sem)
# plt.errorbar(t1, ITC3_Mean_total, yerr=ITC3_sem, color='red')
# figure3 = plt.plot(t1, ITC3_Mean_total, label='')
# plt.xlim([-0.1, 1.1])

ITC3_300ms_avg_total = ITC3_300ms_avg.mean(axis=1)

ITC3_Peak = ITC3_Mean_total[t3 > 0]
print(max(ITC3_Peak))
#print(ITC3_300ms)

# Subplots - All subjects
fig, ax = plt.subplots(3, 1, sharex=True, sharey=True, constrained_layout=True)
ax[0].errorbar(t1, ITC1_Mean_total, yerr=ITC1_sem,color='darkblue', linewidth=2, ecolor='lightsteelblue')
ax[0].set_title('ITC - Gap 16 ms', loc='center', fontsize=12)
ax[1].errorbar(t2, ITC2_Mean_total, yerr=ITC2_sem,color='purple', linewidth=2, ecolor='thistle')
ax[1].set_title('ITC - Gap 32 ms', loc='center', fontsize=12)
ax[2].errorbar(t3, ITC3_Mean_total, yerr=ITC3_sem,color='green', linewidth=2, ecolor='palegreen')
ax[2].set_title('ITC - Gap 64 ms', loc='center', fontsize=12)
plt.xlim([-0.1, 1.1])
plt.ylim([0.02, 0.09])
plt.xlabel('Time (in seconds)')
ax[1].set_ylabel('ITC Value')
#fig.text(-0.03,0.5, 'ITC Value', va='center',rotation ='vertical')
fig.suptitle('ITC for the gap durations all ages (N=27)', x=0.55, fontsize=14)
params = {'legend.fontsize': 'x-large',
          'figure.figsize': (7, 5),
          'ytick.labelsize': 'xx-small',
          'ytick.major.pad': '6'}
plt.rcParams.update(params)
# plt.tight_layout()
#fig.supylabel('ITC Value')
plt.show()
save_loc = ('C:/Users/vmysorea/Desktop/PhD/Stim_Analysis/GapDetection_EEG/Analysis/AnalyzedFiles_Figures/All_Ages/ITC_Figures/')
# plt.savefig(save_loc + 'ITC123_Combined_AllAges.png', dpi=300)

#%% Saving the ITC values of avg upto 300 ms in a matfile
mat_id = {'Subjects':sub,'ITC1_300ms':ITC1_300ms_avg_total, 'ITC2_300ms':ITC2_300ms_avg_total, 'ITC3_300ms':ITC3_300ms_avg_total}
savemat(save_loc + 'ITC_300ms.mat', mat_id)

# %% ITC- Plotting across ages

# %% YNH ITC Plotting

Subjects_YNH = ['S273','S268','S269','S274','S282','S285','S259','S277','S279','S280','S270','S271','S281','S290','S284',
              'S303','S288','S260']  # Load YNH ITC mat data

# ITC_1

ITC1_Mean_Y = np.zeros((len(Subjects_YNH), 5736))
for sub in range(len(Subjects_YNH)):
    subject = Subjects_YNH[sub]
    dat = io.loadmat(data_loc + 'ITC1_' + subject, squeeze_me=True)
    dat.keys()
    itc1 = dat['itc1']
    freqs = dat['freqs']
    n_channels = dat['n_channels']
    t1 = dat['t']

    itc1_avg = itc1[:, :].mean(axis=0)
    ITC1_Mean_Y[sub, :] = itc1_avg

ITC1_Mean_total_Y = ITC1_Mean_Y.mean(axis=0)
ITC1_sem_Y = sem(ITC1_Mean_Y)

tnew1 = t1[t1 > 0]
ITC1_Peak_Y = ITC1_Mean_total_Y[t1 > 0]
print(max(ITC1_Peak_Y))

# ITC 2

ITC2_Mean_Y = np.zeros((len(Subjects_YNH), 5736))
for sub in range(len(Subjects_YNH)):
    subject = Subjects_YNH[sub]
    dat = io.loadmat(data_loc + 'ITC2_' + subject, squeeze_me=True)
    dat.keys()
    itc2 = dat['itc2']
    freqs = dat['freqs']
    ITC2_avg = itc2[:, :].mean(axis=0)
    ITC2_Mean_Y[sub, :] = ITC2_avg

ITC2_Mean_total_Y = ITC2_Mean_Y.mean(axis=0)
ITC2_sem_Y = sem(ITC2_Mean_Y)

tnew1 = t1[t1 > 0]
ITC2_Peak_Y = ITC2_Mean_total_Y[t1 > 0]
print(max(ITC2_Peak_Y))

# ITC_3

ITC3_Mean_Y = np.zeros((len(Subjects_YNH), 5736))
for sub in range(len(Subjects_YNH)):
    subject = Subjects_YNH[sub]
    dat = io.loadmat(data_loc + 'itc3_' + subject, squeeze_me=True)

    dat.keys()
    itc3 = dat['itc3']
    freqs = dat['freqs']
    n_channels = dat['n_channels']
    t1 = dat['t']

    itc3_avg = itc3[:, :].mean(axis=0)
    ITC3_Mean_Y[sub, :] = itc3_avg

ITC3_Mean_total_Y = ITC3_Mean_Y.mean(axis=0)
ITC3_sem_Y = sem(ITC3_Mean_Y)

tnew1 = t1[t1 > 0]
ITC3_Peak_Y = ITC3_Mean_total_Y[t1 > 0]
print(max(ITC3_Peak_Y))

# Subplots across gap durations for YNH
fig, ax = plt.subplots(3, 1, sharex=True, sharey=True, constrained_layout=True)
ax[0].plot(t1, ITC1_Mean_total_Y)
ax[0].errorbar(t1, ITC1_Mean_total_Y, yerr=ITC1_sem_Y, color='darkblue', linewidth=2, ecolor='lightsteelblue')
ax[1].plot(t1, ITC2_Mean_total_Y)
ax[1].errorbar(t1, ITC2_Mean_total_Y, yerr=ITC2_sem_Y, color='purple', linewidth=2, ecolor='thistle')
ax[2].plot(t1, ITC3_Mean_total_Y)
ax[2].errorbar(t1, ITC3_Mean_total_Y, yerr=ITC3_sem_Y, color='green', linewidth=2, ecolor='palegreen')
ax[0].set_title('ITC - Gap 16 ms', loc='center', fontsize=10)
ax[1].set_title('ITC - Gap 32 ms', loc='center', fontsize=10)
ax[2].set_title('ITC - Gap 64 ms', loc='center', fontsize=10)
plt.xlim([-0.1, 1.1])
plt.ylim([0, 0.1])
plt.xlabel('Time (in seconds)')
ax[1].set_ylabel('ITC Value', loc='center')
plt.suptitle('ITC for the gap durations -YNH (N=17)')
plt.rcParams["figure.figsize"] = (5.5, 5)
plt.show()


plt.savefig(save_loc + 'ITC123_Combined_Below35.png', dpi=300)


# %% MNH ITC Plotting

Subjects_MNH = ['S312', 'S078', 'S069', 'S104','S088', 'S072']  # Load MNH ITC mat data

# ITC_1

ITC1_Mean_M = np.zeros((len(Subjects_MNH), 5736))
for sub in range(len(Subjects_MNH)):
    subject = Subjects_MNH[sub]
    dat = io.loadmat(data_loc + 'ITC1_' + subject, squeeze_me=True)

    dat.keys()
    itc1 = dat['itc1']
    freqs = dat['freqs']
    n_channels = dat['n_channels']
    t1 = dat['t']

    itc1_avg = itc1[:, :].mean(axis=0)
    ITC1_Mean_M[sub, :] = itc1_avg

ITC1_Mean_total_M = ITC1_Mean_M.mean(axis=0)
ITC1_sem_M = sem(ITC1_Mean_M)

tnew1 = t1[t1 > 0]
ITC1_Peak_M = ITC1_Mean_total_M[t1 > 0]
print(max(ITC1_Peak_M))

# ITC 2

ITC2_Mean_M = np.zeros((len(Subjects_MNH), 5736))
for sub in range(len(Subjects_MNH)):
    subject = Subjects_MNH[sub]
    dat = io.loadmat(data_loc + 'ITC2_' + subject, squeeze_me=True)

    dat.keys()
    itc2 = dat['itc2']
    freqs = dat['freqs']
    # n_channels=dat['n_channels']
    #t1 = dat['t']

    ITC2_avg = itc2[:, :].mean(axis=0)
    ITC2_Mean_M[sub, :] = ITC2_avg

ITC2_Mean_total_M = ITC2_Mean_M.mean(axis=0)
ITC2_sem_M = sem(ITC2_Mean_M)

tnew1 = t1[t1 > 0]
ITC2_Peak_M = ITC2_Mean_total_M[t1 > 0]
print(max(ITC2_Peak_M))

# ITC_3

ITC3_Mean_M = np.zeros((len(Subjects_MNH), 5736))
for sub in range(len(Subjects_MNH)):
    subject = Subjects_MNH[sub]
    dat = io.loadmat(data_loc + 'itc3_' + subject, squeeze_me=True)

    dat.keys()
    itc3 = dat['itc3']
    freqs = dat['freqs']
    n_channels = dat['n_channels']
    t1 = dat['t']

    itc3_avg = itc3[:, :].mean(axis=0)
    ITC3_Mean_M[sub, :] = itc3_avg

ITC3_Mean_total_M = ITC3_Mean_M.mean(axis=0)
ITC3_sem_M = sem(ITC2_Mean_M)

tnew1 = t1[t1 > 0]
itc3_Peak_M = ITC3_Mean_total_M[t1 > 0]
print(max(itc3_Peak_M))

# Subplots across gap durations for MNH
fig, ax = plt.subplots(3, 1, sharex=True, sharey=True, constrained_layout=True)
ax[0].plot(t1, ITC1_Mean_total_M)
ax[0].errorbar(t1, ITC1_Mean_total_M, yerr=ITC1_sem_M, color='darkblue', linewidth=2, ecolor='lightsteelblue')
ax[1].plot(t1, ITC2_Mean_total_M)
ax[1].errorbar(t1, ITC2_Mean_total_M, yerr=ITC2_sem_M, color='purple', linewidth=2, ecolor='thistle')
ax[2].plot(t1, ITC3_Mean_total_M)
ax[2].errorbar(t1, ITC3_Mean_total_M, yerr=ITC3_sem_M, color='green', linewidth=2, ecolor='palegreen')
ax[0].set_title('ITC - Gap 16 ms', loc='center', fontsize=10)
ax[1].set_title('ITC - Gap 32 ms', loc='center', fontsize=10)
ax[2].set_title('ITC - Gap 64 ms', loc='center', fontsize=10)
plt.xlim([-0.1, 1.1])
plt.xlabel('Time (in seconds)')
ax[1].set_ylabel('ITC Value', loc='center')
plt.suptitle('ITC for the gap durations -MNH (N=6)')
plt.rcParams["figure.figsize"] = (5.5, 5)
plt.savefig(save_loc + 'ITC123_Combined_MNH.png', dpi=300)

# %% ONH ITC Plotting

Subjects_ONH = ['S312','S347','S340','S078','S069', 'S088','S072','S308','S344','S105','S291','S310','S339']

ITC1_Mean_O = np.zeros((len(Subjects_ONH), 5736))
for sub in range(len(Subjects_ONH)):
    subject = Subjects_ONH[sub]
    dat = io.loadmat(data_loc + 'ITC1_' + subject, squeeze_me=True)

    dat.keys()
    itc1 = dat['itc1']
    freqs = dat['freqs']
    n_channels = dat['n_channels']
    t1 = dat['t']

    itc1_avg = itc1[:, :].mean(axis=0)
    ITC1_Mean_O[sub, :] = itc1_avg

ITC1_Mean_total_O = ITC1_Mean_O.mean(axis=0)
ITC1_sem_O = sem(ITC1_Mean_O)

tnew1 = t1[t1 > 0]
ITC1_Peak_O = ITC1_Mean_total_O[t1 > 0]
print(max(ITC1_Peak_O))

# ITC_2

ITC2_Mean_O = np.zeros((len(Subjects_ONH), 5736))
for sub in range(len(Subjects_ONH)):
    subject = Subjects_ONH[sub]
    dat = io.loadmat(data_loc + 'ITC2_' + subject, squeeze_me=True)
    dat.keys()
    itc2 = dat['itc2']
    freqs = dat['freqs']
    # n_channels=dat['n_channels']
    t1 = dat['t']

    ITC2_avg = itc2[:, :].mean(axis=0)
    ITC2_Mean_O[sub, :] = ITC2_avg

ITC2_Mean_total_O = ITC2_Mean_O.mean(axis=0)
ITC2_sem_O = sem(ITC2_Mean_O)

tnew1 = t1[t1 > 0]
ITC2_Peak_O = ITC2_Mean_total_O[t1 > 0]
print(max(ITC2_Peak_O))

# ITC 3

ITC3_Mean_O = np.zeros((len(Subjects_ONH), 5736))
for sub in range(len(Subjects_ONH)):
    subject = Subjects_ONH[sub]
    dat = io.loadmat(data_loc + 'itc3_' + subject, squeeze_me=True)

    dat.keys()
    itc3 = dat['itc3']
    freqs = dat['freqs']
    n_channels = dat['n_channels']
    t1 = dat['t']

    itc3_avg = itc3[:, :].mean(axis=0)
    ITC3_Mean_O[sub, :] = itc3_avg

ITC3_Mean_total_O = ITC3_Mean_O.mean(axis=0)
ITC3_sem_O = sem(ITC3_Mean_O)

tnew1 = t1[t1 > 0]
itc3_Peak_O = ITC3_Mean_total_O[t1 > 0]
print(max(itc3_Peak_O))

# Subplots across gap durations for ONH
fig, ax = plt.subplots(3, 1, sharex=True, sharey=True, constrained_layout=True)
ax[0].plot(t1, ITC1_Mean_total_O)
ax[0].errorbar(t1, ITC1_Mean_total_O, yerr=ITC1_sem_O, color='darkblue', linewidth=2, ecolor='lightsteelblue')
ax[1].plot(t1, ITC2_Mean_total_O)
ax[1].errorbar(t1, ITC2_Mean_total_O, yerr=ITC2_sem_O, color='purple', linewidth=2, ecolor='thistle')
ax[2].plot(t1, ITC3_Mean_total_O)
ax[2].errorbar(t1, ITC3_Mean_total_O, yerr=ITC3_sem_O,color='green', linewidth=2, ecolor='palegreen')
ax[0].set_title('ITC - Gap 16 ms', loc='center', fontsize=10)
ax[1].set_title('ITC - Gap 32 ms', loc='center', fontsize=10)
ax[2].set_title('ITC - Gap 64 ms', loc='center', fontsize=10)
plt.ylim([0, 0.1])
plt.xlim([-0.1, 1.1])
plt.xlabel('Time (in seconds)')
ax[1].set_ylabel('ITC Value', loc='center')
plt.suptitle('ITC for the gap durations - >35y (N=10)')
plt.rcParams["figure.figsize"] = (5.5, 5)
plt.show()
plt.savefig(save_loc + 'ITC123_Combined_Above35.png', dpi=300)

# %% Subplots across ages

# ax[0].plot(t1, ITC1_Mean_total_Y)
x = ITC1_Mean_total_Y/ITC1_Mean_total_Y+ITC2_Mean_total_Y+ITC3_Mean_total_Y
x1=ITC1_sem_Y/ITC1_sem_Y+ITC2_sem_Y+ITC3_sem_Y
y=ITC1_Mean_total_O/ITC1_Mean_total_O+ITC2_Mean_total_O+ITC3_Mean_total_O
y1=ITC1_sem_O/ITC1_sem_O+ITC2_sem_O+ITC3_sem_O

a= ITC2_Mean_total_Y/ITC1_Mean_total_Y+ITC2_Mean_total_Y+ITC3_Mean_total_Y
a1=ITC2_sem_Y/ITC1_sem_Y+ITC2_sem_Y+ITC3_sem_Y
b=ITC2_Mean_total_O/ITC1_Mean_total_O+ITC2_Mean_total_O+ITC3_Mean_total_O
b1=ITC2_sem_O/ITC1_sem_O+ITC2_sem_O+ITC3_sem_O

c=ITC3_Mean_total_Y/ITC1_Mean_total_Y+ITC2_Mean_total_Y+ITC3_Mean_total_Y
c1=ITC3_sem_Y/ITC1_sem_Y+ITC2_sem_Y+ITC3_sem_Y
d=ITC3_Mean_total_O/ITC1_Mean_total_O+ITC2_Mean_total_O+ITC3_Mean_total_O
d1=ITC3_sem_O/ITC1_sem_O+ITC2_sem_O+ITC3_sem_O


fig, ax = plt.subplots(3, 1, sharex=True, sharey=True, constrained_layout=True)
ax[0].errorbar(t1, x, yerr=x1*0.1,color='green', linewidth=2, ecolor='lightgreen',label='Below 35y (N=19)')
# ax[0].plot(t1, y)
ax[0].errorbar(t1, y, yerr=y1*0.1,color='purple', linewidth=2, ecolor='thistle',label='Above 35y (N=14)')
ax[0].set_title('ITC - Gap 16 ms', loc='center', fontsize=10)
# ax[1].plot(t1, x, t1, y)
ax[1].errorbar(t1, a, yerr=a1*0.1, color='green', linewidth=2, ecolor='lightgreen')
ax[1].errorbar(t1, b, yerr=b1*0.1,color='purple', linewidth=2, ecolor='thistle')
ax[1].set_title('ITC - Gap 32 ms', loc='center', fontsize=10)
# ax[2].plot(t1, a, t1, b)
ax[2].errorbar(t1, c, yerr=c1*0.1, color='green', linewidth=2, ecolor='lightgreen')
ax[2].errorbar(t1, d, yerr=d1*0.1, color='purple', linewidth=2, ecolor='thistle')
ax[2].set_title('ITC - Gap 64 ms', loc='center', fontsize=10)
plt.xlim([-0.1, 0.5])
plt.ylim([0.5,2])
ax[0].legend(prop={'size': 6})
plt.xlabel('Time (in seconds)')
fig.text(0.0001, 0.5, 'ITC Value', va='center', rotation='vertical')
plt.suptitle('ITC for the gap durations')
plt.rcParams["figure.figsize"] = (5.5, 5)
plt.tight_layout()
plt.show()

plt.savefig(save_loc + 'ITC123_Below,Above35_0.5.png', dpi=300)