%% SGH EEG Staging Preparation
% This script is to prepare PSG signals as sleep stage segments for
% training and testing sleep staging algorithms
%
% tasks performed: 
% 1. downsample from 256 Hz to 128 Hz
% 2. remove epochs with "unknow" / "NaN" labels

clear;
clc;

%% Prepare Environment Variables

load('psg_data/edf_path.mat');

no_train = length(train_path);
no_test = length(test_path);
no_valid = length(valid_path);

PSG_sampling_freq = 256;
epoch_length = 30;
sample_p_epoch_ori = PSG_sampling_freq * epoch_length;
decimation_freq = 128;
decimation_ratio = PSG_sampling_freq / decimation_freq; 
sample_p_epoch_new = decimation_freq * epoch_length;

% EEG: C3-M2, O1-M2, EOG: E1-M2, EMG: chin1-chin2
selected_channels = [1, 3, 5, 7];

% threshold for channels 
tEEG = 70; % unit uV
tEOG = 70; % unit uV
rEMG = 2;  % unit mV, ratio of EMG

% total number of sleep stages: W, N1, N2, N3, REM
num_class = 5;

% ============ VERY IMPORTANT ===============
% epoch offset 
% the first few epochs are usually for sensor callibration purpose
% even though labels as WAKE
% so remove the first few epochs for all subject data
% here 10 epochs means 5 minutes
% ===========================================
epoch_offset = 10;


%% Prepare Train PSG Data
% load train data
load('train_edf_data.mat');

train_data_collect.data = [];
train_data_collect.label = [];

epoch_iter = 1;

for train_i = 1:no_train
    
    edf_path = ['psg_data/', train_path{train_i}];
    labels = parse_sleep_stage(edf_path);
    no_epochs = length(labels);
    
    train_data_collect.data = [train_data_collect.data; ...
                                zeros(no_epochs, 4, sample_p_epoch_new)];
    train_data_collect.label = [train_data_collect.label; ...
                                zeros(no_epochs, num_class)];
    
    for epoch_i = 1:no_epochs
        psg_raw = train_edf_data{train_i, 2}(selected_channels, ...
            ((epoch_i-1)*sample_p_epoch_ori+1):(epoch_i*sample_p_epoch_ori));
        label = labels(epoch_i, 1);
        
        % remove epochs with "unknown" label
        if ~isnan(label)
            temp = downsample(psg_raw.', decimation_ratio).';
            train_data_collect.data(epoch_iter, :, :) = raw_data_thresholding(temp, tEEG, tEOG, rEMG);
            train_data_collect.label(epoch_iter, :) = one_hot_with_total_num(label, num_class);
        end
        epoch_iter = epoch_iter + 1;
    end
end

% perform offset data to remove possible epochs for calibration purpose at
% the beginning
train_data_collect.data(1:epoch_offset, :, :) = [];
train_data_collect.label(1:epoch_offset, :) = [];

% interchange dimension, from [epoch, channel, data] to [epoch_i, data, channel]
% to be consistent in Python environment
train_data_collect.data = permute(train_data_collect.data, [1, 3, 2]);

clear('train_edf_data', 'train_i', 'train_path', 'no_train');
save('train_data_PSG.mat', 'train_data_collect', '-v7.3');
clear('train_data_collect');

%% Prepare Test EEG data
% load test data
load('test_edf_data.mat');

test_data_collect.data = [];
test_data_collect.label = [];

epoch_iter = 1;

for test_i = 1:no_test
    
    edf_path = ['psg_data/', test_path{test_i}];
    labels = parse_sleep_stage(edf_path);
    no_epochs = length(labels);
    
    test_data_collect.data = [test_data_collect.data; ...
                                zeros(no_epochs, 4, sample_p_epoch_new)];
    test_data_collect.label = [test_data_collect.label; ...
                                zeros(no_epochs, num_class)];
    
    for epoch_i = 1:no_epochs
        psg_raw = test_edf_data{test_i, 2}(selected_channels, ...
            ((epoch_i-1)*sample_p_epoch_ori+1):(epoch_i*sample_p_epoch_ori));
        label = labels(epoch_i, 1);
        
        % remove epochs with "unknown" label
        if ~isnan(label)
            temp = downsample(psg_raw.', decimation_ratio).';
            test_data_collect.data(epoch_iter, :, :) = raw_data_thresholding(temp, tEEG, tEOG, rEMG);
            test_data_collect.label(epoch_iter, :) = one_hot_with_total_num(label, num_class);
        end
        epoch_iter = epoch_iter + 1;
    end
end

% perform offset data to remove possible epochs for calibration purpose at
% the beginning
test_data_collect.data(1:epoch_offset, :, :) = [];
test_data_collect.label(1:epoch_offset, :) = [];

% interchange dimension, from [epoch, channel, data] to [epoch_i, data, channel]
% to be consistent in Python environment
test_data_collect.data = permute(test_data_collect.data, [1, 3, 2]);

clear('test_edf_data', 'test_i', 'test_path', 'no_test');
save('test_data_PSG.mat', 'test_data_collect', '-v7.3');
clear('test_data_collect');

%% Prepare valid EEG data
% load valid data
load('valid_edf_data.mat');

valid_data_collect.data = [];
valid_data_collect.label = [];

epoch_iter = 1;

for valid_i = 1:no_valid
    
    edf_path = ['psg_data/', valid_path{valid_i}];
    labels = parse_sleep_stage(edf_path);
    no_epochs = length(labels);
    
    valid_data_collect.data = [valid_data_collect.data; ...
                                zeros(no_epochs, 4, sample_p_epoch_new)];
    valid_data_collect.label = [valid_data_collect.label; ...
                                zeros(no_epochs, num_class)];
    
    for epoch_i = 1:no_epochs
        psg_raw = valid_edf_data{valid_i, 2}(selected_channels, ...
            ((epoch_i-1)*sample_p_epoch_ori+1):(epoch_i*sample_p_epoch_ori));
        label = labels(epoch_i, 1);
        
        % remove epochs with "unknown" label
        if ~isnan(label)
            temp = downsample(psg_raw.', decimation_ratio).';
            valid_data_collect.data(epoch_iter, :, :) = raw_data_thresholding(temp, tEEG, tEOG, rEMG);
            valid_data_collect.label(epoch_iter, :) = one_hot_with_total_num(label, num_class);
        end
        epoch_iter = epoch_iter + 1;
    end
end

% perform offset data to remove possible epochs for calibration purpose at
% the beginning
valid_data_collect.data(1:epoch_offset, :, :) = [];
valid_data_collect.label(1:epoch_offset, :) = [];

% interchange dimension, from [epoch, channel, data] to [epoch_i, data, channel]
% to be consistent in Python environment
valid_data_collect.data = permute(valid_data_collect.data, [1, 3, 2]);

clear('valid_edf_data', 'valid_i', 'valid_path', 'no_valid');
save('valid_data_PSG.mat', 'valid_data_collect', '-v7.3');
clear('valid_data_collect');

clear('edf_path', 'epoch_i', 'epoch_iter', 'epoch_length', ...
      'label', 'no_epochs', 'psg_raw', 'temp');
