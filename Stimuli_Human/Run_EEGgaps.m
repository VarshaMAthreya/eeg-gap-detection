%%This code runs the EEG Gap Detection Paradigm with three gap durations - 16, 32, 64 ms. 
clear all; close all hidden; clc; %#ok<CLALL>

%% Subject Info
subj = input('Please subject ID:', 's');

try
    %% Setting random generator seed and state
    load('s.mat');
    rng(s);
    
    %% Setting up sound card
    fig_num=99;
    USB_ch=1;
    IAC = -1;
    FS_tag = 3;
    Fs = 48828.125;
    [f1RZ,RZ,FS]=load_play_circuit(FS_tag,fig_num,USB_ch,0,IAC);
    
    %% Experiment parameters
    ntrials = 250; % Per gap
    level = 78; % Target level
    SNR = 10; % RMS-based SNR for tone in noise
    gaps = [16, 32, 64] * 1e-3; %Setting gap durations
    fc = 4000;
    rampgap = 0.001;
    rampoverall = 0.010;
    markerdur = 0.5; % Each tonal marker before gap
    ngaps = 3; % Three gaps per trial
    isi = 0.5;
        
    % Pause Off (Start saving EEG)
    invoke(RZ, 'SetTagVal', 'trigval',253);
    invoke(RZ, 'SoftTrg', 6);
    
    WaitSecs(2.0);
    stimrms = 0.1; % Set by makeGDTstim_tone.m
        
    % Using jitter to make sure that background noise averages out across
    % trials. We use jitter that is random between 0 and 100ms (0.1s). So
    % average duration added by the jitter is 50 ms (0.05s).
    jitlist = rand(ntrials, numel(gaps))*0.1;
        
    tstart = tic;
    for j = 1:ntrials
        for c = 1:numel(gaps)
            gap = gaps(c);
            x = makeGDTstim_tone(fc, SNR, gap, markerdur, ngaps,...
                rampgap, rampoverall, Fs);
 
            chanL = x;
            chanR = x;
            
            stimTrigger = c;
            
            jit = jitlist(j, c);
            
            stimlength = numel(x);
            dur = stimlength/Fs;
            
            %-----------------
            % Why 111 for ER-2?
            %-----------------
            % ER-2s give about 100dB SPL for a 1kHz tone with a 1V-rms drive.
            % Max output is +/-5V peak i.e 3.54V-rms which is 11 dB higher.
            % Thus 111 dB-SPL is the sound level for tones when they occupy full
            % range.
            
            % Full range in MATLAB for a pure tone is +/- 1 which is 0.707 rms and
            % that corresponds to 111 dB SPL at the end. So if we want a signal
            % with rms sigrms to be x dB, then (111 - x) should be
            % db(sigrms/0.707).
            
            
            dropL = 111 - level + 3 + db(stimrms); % The extra 3 for the 0.707 factor
            
            dropR = 111 - level + 3 + db(stimrms);
            
            
            invoke(RZ, 'SetTagVal', 'trigval', stimTrigger);
            invoke(RZ, 'SetTagVal', 'nsamps', stimlength);
            invoke(RZ, 'WriteTagVEX', 'datainL', 0, 'F32', chanL); %write to buffer left ear
            invoke(RZ, 'WriteTagVEX', 'datainR', 0, 'F32', chanR); %write to buffer right ear
            
            invoke(RZ, 'SetTagVal', 'attA', dropL); %setting analog attenuation L
            invoke(RZ, 'SetTagVal', 'attB', dropR); %setting analog attenuation R
            
            WaitSecs(0.05); % Just giving time for data to be written into buffer
            %Start playing from the buffer:
            invoke(RZ, 'SoftTrg', 1); %Playback trigger
            fprintf(1,' Trial Number %d/%d\n', j, ntrials);
            WaitSecs(dur + isi + jit);
            
        end
    end
    toc(tstart);
    %Clearing I/O memory buffers:
    invoke(RZ,'ZeroTag','datainL');
    invoke(RZ,'ZeroTag','datainR');
    WaitSecs(3.0);
    
    % Pause On (Stop saving EEG) 
    invoke(RZ, 'SetTagVal', 'trigval', 254);
    invoke(RZ, 'SoftTrg', 6);
    
    close_play_circuit(f1RZ,RZ);
    fprintf(1,'\n Done with data collection!\n');
    
catch me
    close_play_circuit(f1RZ,RZ);
    rethrow(me);
end

