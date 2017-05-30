%% SGH EEG Staging Preparation
% This script is to prepare EEG signals as sleep stage segments for
% training and testing sleep staging algorithms
%
% tasks performed: 
% 1. downsample from 256 Hz to 128 Hz
% 2. remove epochs with "unknow" / "NaN" labels


clear;
clc;

%% Prepare Environment Variables

load('edf_path.mat');

no_train = length(train_path);
no_test = length(test_path);
no_valid = length(valid_path);

EEG_sampling_freq = 256;
epoch_length = 30;
sample_p_epoch_ori = EEG_sampling_freq * epoch_length;
decimation_freq = 128;
decimation_ratio = EEG_sampling_freq / decimation_freq; 
sample_p_epoch_new = decimation_freq * epoch_length;


%% Prepare Train EEG Data
% load train data
load('train_edf_data.mat');

train_data_collect.data = [];
train_data_collect.label = [];

epoch_iter = 1;

for train_i = 1:no_train
    
    edf_path = train_path{train_i};
    labels = parse_sleep_stage(edf_path);
    no_epochs = length(labels);
    
    train_data_collect.data = [train_data_collect.data; ...
                                zeros(no_epochs, 6, sample_p_epoch_new)];
    
    for epoch_i = 1:no_epochs
        eeg_raw = train_edf_data{train_i, 2}(1:6, ...
            ((epoch_i-1)*sample_p_epoch_ori+1):(epoch_i*sample_p_epoch_ori));
        label = labels(epoch_i, 1);
        
        % remove epochs with "unknown" label
        if ~isnan(label)
            train_data_collect.data(epoch_iter, :, :) = downsample(eeg_raw.', decimation_ratio).';
            train_data_collect.label(epoch_iter) = label + 1;
        end
        epoch_iter = epoch_iter + 1;
    end
end


clear('train_edf_data', 'train_i', 'train_path', 'no_train');

%% Prepare Test EEG data
% load test data
load('test_edf_data.mat');

test_data_collect.data = [];
test_data_collect.label = [];

epoch_iter = 1;

for test_i = 1:no_test
    
    edf_path = test_path{test_i};
    labels = parse_sleep_stage(edf_path);
    no_epochs = length(labels);
    
    test_data_collect.data = [test_data_collect.data; ...
                                zeros(no_epochs, 6, sample_p_epoch_new)];
    
    for epoch_i = 1:no_epochs
        eeg_raw = test_edf_data{test_i, 2}(1:6, ...
            ((epoch_i-1)*sample_p_epoch_ori+1):(epoch_i*sample_p_epoch_ori));
        label = labels(epoch_i, 1);
        
        % remove epochs with "unknown" label
        if ~isnan(label)
            test_data_collect.data(epoch_iter, :, :) = downsample(eeg_raw.', decimation_ratio).';
            test_data_collect.label(epoch_iter) = label + 1;
        end
        epoch_iter = epoch_iter + 1;
    end   
end
clear('test_edf_data', 'test_i', 'test_path', 'no_test');

%% Prepare valid EEG data
% load validation data
load('valid_edf_data.mat');

valid_data_collect.data = [];
valid_data_collect.label = [];

epoch_iter = 1;

for valid_i = 1:no_valid
    
    edf_path = valid_path{valid_i};
    labels = parse_sleep_stage(edf_path);
    no_epochs = length(labels);
    
    valid_data_collect.data = [valid_data_collect.data; ...
                                zeros(no_epochs, 6, sample_p_epoch_new)];
    
    for epoch_i = 1:no_epochs
        eeg_raw = valid_edf_data{valid_i, 2}(1:6, ...
            ((epoch_i-1)*sample_p_epoch_ori+1):(epoch_i*sample_p_epoch_ori));
        label = labels(epoch_i, 1);
        
        % remove epochs with "unknown" label
        if ~isnan(label)
            valid_data_collect.data(epoch_iter, :, :) = downsample(eeg_raw.', decimation_ratio).';
            valid_data_collect.label(epoch_iter) = label + 1;
        end
        
        epoch_iter = epoch_iter + 1;
    end
end
clear('valid_edf_data', 'valid_i', 'valid_path', 'no_valid', 'eeg_raw', 'labels');

save('SGH_EEG_data_prepared.mat', 'train_data_collect', ...
     'test_data_collect', 'valid_data_collect', 'decimation_freq', '-v7.3');
