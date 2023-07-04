
function x = GetChinGDTstim_tone(fc, SNR, gapdur, markerdur, ngaps, rampgap, rampoverall, fs)
% Stimulus for Chin GDT EEG experiment
%
% USAGE:
%   x = GetChinGDTstim_tone(fc, gapdur, markerdur, ngaps, rampgap, rampoverall,
%   fs);
% INPUTS:
%   fc - Center frequency of tone (and noise band) in Hz
%   SNR - Tone-to-octavebandnoise ratio (dB)
%   gapdur - Gap duration in seconds
%   markerdur - Duration of each tonal marker (i.e., between gaps) in
%   seconds
%   ngaps - Number of gaps per trial
%   rampgap - Ramp for the tonal markers before and after each gap
%   (seconds)
%   rampoverall - Ramp for overall onset and offset (seconds)
%   fs - Sampling rate (Hz)
%
% OUTPUTS:
%   x - Mono signal with target tone RMS of 0.1 and octave-band noise at
%   the desired SNR.
%
%-----------------
% Make silent gap
gap = zeros(round(gapdur * fs), 1);
gap = gap(:); % Make it column vector, just to be sure

% Make tonal markers
t = 0:(1/fs):(markerdur - 1/fs);
tone = rampsound(sin(2*pi*fc*t), fs, rampgap);
tone = tone(:);  % Make it column vector

sig = tone;
for k = 1:ngaps
    sig = [sig; gap; tone]; %#ok<AGROW>
end
    

% Make noise token for the whole duration
bw = 1; % Half octave above and below
fc_noise = fc;
noisedur = numel(sig) / fs;
noise = makeEqExNoiseFFT(bw, fc_noise, noisedur, fs, rampgap, 0);

x = noise * db2mag(-SNR)/rms(noise) + sig/rms(sig);
x = rampsound(x, fs, rampoverall) * 0.1;

