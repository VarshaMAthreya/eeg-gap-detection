import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat, wavfile

# Set chunk size for large plots
plt.rcParams['agg.path.chunksize'] = 10000

# Load the .mat file
mat_path = 'C:/Users/varsh/Desktop/PhD/FinalPaper/Figures/EEG/Stim/GDT_HumanStim_250.mat'
data = loadmat(mat_path, squeeze_me=True)

# Get sampling rate
fs = int(data['Fs'])  # Ensure it's an integer for plotting and WAV writing

# Get waveform
x = data['x']
x = np.asarray(x, dtype=np.float32)  # Convert to float32 for WAV compatibility

# Create time vector
t = np.arange(len(x)) / fs

# Plotting
plt.figure(figsize=(10, 4))
plt.plot(t, x, color='dimgrey')
plt.axvline(0.564, linestyle='dashed', color='blue', linewidth=3)
plt.axvline(1.128, linestyle='dashed', color='blue', linewidth=3)
plt.axvline(1.692, linestyle='dashed', color='blue', linewidth=3)
plt.text(0.45, 0.3, 'Trigger', fontsize=18, color='blue', weight='bold')
plt.xlabel('Time (in s)', fontsize=22)
plt.ylabel('Stim Amplitude', fontsize=22)
plt.xticks(fontsize=20)
# plt.xlim(0,2)
plt.yticks([], [])  # Hide y-axis ticks
plt.tight_layout()
plt.show()

# Save as WAV file
# wavfile.write('C:/Users/varsh/Desktop/PhD/FinalPaper/Figures/EEG/Stim/GDTHuman_64ms.wav', fs, x)

# Save figure (Note: put after plt.show() if you want to see it before saving)
plt.savefig('C:/Users/varsh/Desktop/PhD/FinalPaper/Figures/EEG/Stim/GDTHuman_64ms.png', dpi=500, transparent=True)
