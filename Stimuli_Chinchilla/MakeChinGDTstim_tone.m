%% Make stim for Chinchilla EEG GDT Experiment
%% Parameters
fs = 48828.125;
ntrials = 700; % Per gap
level = 78; % Target level
SNR = 10; % RMS-based SNR for tone in noise
gaps = [16, 32, 64] * 1e-3; %Setting gap durations - Trying with the same as humans 
fc = 4000;
rampgap = 0.001;
rampoverall = 0.010;
markerdur = 1.0; % Each tonal marker before gap
ngaps = 1; % One gap per trial
isi = 0.5;

stimrms = 0.1; % Set by GetChinGDTstim_tone.m
jitlist = rand(ntrials, numel(gaps))*0.1;
%conds = repmat([1, 2, 3],1,ntrials);
% conds = conds(randperm(length(conds)));
%% Generate Stims
y = [];
stims=[];
t=[];
trigs=[];

for j = 1:ntrials
    for c = 1:numel(gaps)
        gap = gaps(c);
        x = GetChinGDTstim_tone(fc, SNR, gap, markerdur, ngaps,...
                rampgap, rampoverall, fs);
      
%         chanL = x;
%         chanR = x;
        %jit = jitlist(j, c);
%         stimlength = numel(x);
%         dur = stimlength/fs;

        stimTrigger = c;
        y{c}=x;
        y1=vertcat(y{:});
        t{c}=stimTrigger;
        t1=vertcat(t{:});
        %t=cell2mat(t);
    end
    gdt_trigs{j}=t1;
    stims{j}=y1;
    
    %stims = cell2mat(stims);
end
%% Save Stuff
save('ChinGDTStims_1s_700.mat','stims','gdt_trigs', 'fs')