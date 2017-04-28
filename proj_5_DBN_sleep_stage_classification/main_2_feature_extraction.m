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


%% Extract features
featurenames = {'EEG delta' 'EEG theta' 'EEG alpha' 'EEG beta' 'EEG gamma' ...
    'EOG delta' 'EOG theta' 'EOG alpha' 'EOG beta' 'EOG gamma' 'EMG delta' ...
    'EMG theta' 'EMG alpha' 'EMG beta' 'EMG gamma' 'EMG median' 'EOG corr' ...
    'EEG kurtosis' 'EOG kurtosis' 'EMG kurtosis' 'EOG std' 'EEG entropy' ...
    'EOG entropy' 'EMG entropy' 'EEG spectral mean' 'EOG spectral mean' ...
    'EMG spectral mean' 'EEG fractal exponent'};
E = cell(1,nfiles);
truestate = cell(1,nfiles);
for i = 1 : nfiles
    fprintf('Extracting features... File %i of %i\n', i, nfiles);
    load(['.\data\p' filek{i} '.mat']);
    E{i} = sect2E(s, HDR.SampleRate, HDR.SampleRate);
    truestate{i} = downsample(v, HDR.SampleRate);
end

save Eucddb E truestate featurenames
