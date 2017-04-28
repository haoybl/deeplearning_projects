%% Introduction
% This is the main file for duplicating the results in
% "Sleep Stage Classification using Unsupervised Feature Learning"
% by Martin Längkvist, Lars Karlsson and Amy Loutfi
%
% This package and the paper can be downloaded from: aass.oru.se\~mlt

%% Initialize
% This section intitializes parameters

clear all

cd('D:\OneDrive\NTU\PhD\PHD DATA\projects\proj_5_DBN_sleep_stage_classification') % set matlab path to installation path\sleep\
filek = {'02' '03' '05' '06' '07' '08' '09' '10' '11' '12' '13' '14' '15' '17' '18' '19' '20' '21' '22' '23' '24' '25' '26' '27' '28'}; %uncomment if all night recordings were downloaded
%filek = {'02' '03' '05' '06' '07'}; % If only the first 5 night recordings were downloaded
nfiles = length(filek);

%% Load and preprocess data and labels. Save to .mat files
prepath = './data/ucddb0';
for k=1:nfiles
    fprintf('Loading data... File %i of %i\n', k, nfiles);
    x=cell(1,4); Fs=cell(1,4); Label=cell(1,4); Dimension=cell(1,4); Coef = cell(1,4);
    HDR=[];
    for i = 1:4
        [x{i}, Fs{i}, Start_date, Start_time, Label{i}, Dimension{i}, Coef{i}, Nmb_chans, N] = readedf_2([prepath filek{k} '.rec'], i-1, 0, inf );
    end
    s = [x{4} interp(x{1},2) interp(x{2},2) interp(x{3},2)]; %interp to upsample EOG and EMG signals
    clear x
    HDR.Label = {Label{4} Label{1} Label{2} Label{3}};
    HDR.SampleRate = Fs{4};
    HDR.datafile = [prepath filek{k} '.rec'];
    HDR.EEG = 1;
    HDR.EOG = 2:3;
    HDR.EMG = 4;
    HDR.simtime = rows(s)/HDR.SampleRate;
    
    % Notch filter
    w0 = 50*(2/HDR.SampleRate);
    [b,a] = iirnotch(w0, w0/35);
    s = filtfilt(b, a, s);
    
    % Bandpass filter
    s(:,1) = bpfilter(s(:,1), [0.3 HDR.SampleRate/2]);
    s(:,2:3) = bpfilter(s(:,2:3), [0.3 HDR.SampleRate/2]);
    s(:,4) = bpfilter(s(:,4), [10 HDR.SampleRate/2]);
    
    % Downsample by 2
    s = s(1:2:end,:);
    HDR.SampleRate = HDR.SampleRate/2;
    
    % Replace NaN and Inf with 0
    s(isnan(s)) = 0;
    s(isinf(s)) = 0;
    
    % Convert to single
    s = single(s);
    
    % Load annotations. Change to my liking.
    v = load([prepath filek{k} '_stage.txt']);
    v = single(myupsample(v, 30*HDR.SampleRate));
    temp = v;
    v(temp==0)=5; % Wake
    v(temp==1)=4; % REM
    v(temp==2)=3; % Stage 1
    v(temp==3)=2; % Stage 2
    v(temp==4)=1; % Stage 3
    v(temp==5)=1; % Stage 4
    v(temp==6)=0; % Artifact
    v(temp==7)=0; % Indeterminate
    
    % Make s and v equal length
    [s, v] = mybalance(s, v);
    
    save(['./data/p' filek{k}],'s', 'HDR', 'v')
    clear s v temp HDR
end