function x = makeGDTstim_OLD(gapdur, dur, pos, ramp, fs)
% Stimulus for GDT EEG experiment
%
% USAGE:
%   x = makeGDTstim(gapdur, dur, pos, fs);
% INPUTS:
%   gapdur - Gap duration in seconds
%   dur - Total stimulus duration in seconds
%   pos - Position of the gap within the stimulus (seconds from start)
%   ramp - Ramp duration for onset/offset and the gap
%   fs - Sampling rate (Hz)
%
% OUTPUTS:
%   x - Mono signal with RMS of 0.1 with the desired parameters
%
%-----------------

bw = 1.415;
fc_noise = 4899;

% Make noise token part to precede the gap
noise1 = makeEqExNoiseFFT(bw, fc_noise, pos, fs, ramp, 0);

% Make silent gap
gap = zeros(round(gapdur * fs), 1);

% Make noise token part to precede the gap
noise2 = makeEqExNoiseFFT(bw, fc_noise, (dur - pos - gapdur), fs, ramp, 0);

x = [noise1; gap; noise2];

