gaps = [16, 32, 64] * 1e-3;
gaps = gaps(:).';
fc = 4000;
rampgap = 0.001;
rampoverall = 0.010;
markerdur = 0.5; % Each tonal marker before gap
ngaps = 1; % Three gaps per trial
isi = 0.5;
Fs = 48828.125;
SNR=10;
ntrials=10;

%% Make stims 
y = {};
stims ={};
for j = 1:ntrials
    for c = 1:numel(gaps)
        %gaps(:).'
        gapdur = gaps(c);
        x = makeGDTstim_tone(fc, SNR,gapdur, markerdur, ngaps,...
                rampgap, rampoverall, Fs);
        y{c}=x;
    end
    stims{j}=y;
end
save('GDT_HumanStim_1.mat','stims','Fs')

