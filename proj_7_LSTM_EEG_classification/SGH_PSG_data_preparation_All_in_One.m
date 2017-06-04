%%  SGH PSG data preparation All in One
%
% This script is used to prepare SGH PSG data for use in Python environment

% -----------------------------------------------
%               IMPORTANT !!!!
% -----------------------------------------------
% Data not usefull:
% 024.edf, 034.edf, 033.edf ...

clear;
clc;

%% Define Environmental Variables & prepare file path
no_edf = 20;
no_train = 16;
no_test = 2;
no_valid = 2;

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
epoch_begin_offset = 10;
epoch_end_offset = 20;
p = randperm(no_edf);

for i = 1:no_train
    train_path{i} = [num2str(p(i), '%03d'), '.edf']; %#ok<*SAGROW>
end

for i = 1:no_test
    test_path{i} = [num2str(p(no_train+i), '%03d'), '.edf'];
end

for i = 1:no_valid
    valid_path{i} = [num2str(p(no_train+no_test+i), '%03d'), '.edf'];
end

save('edf_path.mat', 'train_path', 'test_path', 'valid_path');

%% Read Train data
% Train data
train_data_collect.data = [];
train_data_collect.label = [];

epoch_iter = 0;

no_train = length(train_path);

for train_i = 1:no_train
    % Read Raw Data
    edf_path = ['psg_data/', train_path{train_i}];
    if ~exist(edf_path, 'file')
        disp(edf_path);
        disp('File not Found. Continue to Next loop...');
        continue;
    end
    [hdr, record] = edfread(edf_path);
    record = record(selected_channels, :);
    % Read labels
    labels = parse_sleep_stage(edf_path);
    no_epochs = length(labels);
    
    train_data_collect.data = [train_data_collect.data; ...
                                zeros(no_epochs-epoch_begin_offset-epoch_end_offset, 4, sample_p_epoch_new)];
    train_data_collect.label = [train_data_collect.label; ...
                                zeros(no_epochs-epoch_begin_offset-epoch_end_offset, num_class)];
                            
    % Use offset to remove beginning few epochs for calibration purposes
    for epoch_i = epoch_begin_offset+1:no_epochs-epoch_end_offset-1
        psg_raw = record(:, ...
            ((epoch_i-1)*sample_p_epoch_ori+1):(epoch_i*sample_p_epoch_ori));
        label = labels(epoch_i, 1);
        
        % remove epochs with "unknown" label
        if ~isnan(label)
            epoch_iter = epoch_iter + 1;
            temp = downsample(psg_raw.', decimation_ratio).';
            train_data_collect.data(epoch_iter, :, :) = raw_data_thresholding(temp, tEEG, tEOG, rEMG);
            train_data_collect.label(epoch_iter, :) = one_hot_with_total_num(label, num_class);
        end
    end
    
    
    % clear extra zeros epochs
    extra = size(train_data_collect.data, 1);
    if extra > epoch_iter
        train_data_collect.data(epoch_iter+1:end, :, :) = [];
        train_data_collect.label(epoch_iter+1:end, :) = [];
    end
end

% Generate histogram of classes to check class imbalance
class_counter(train_data_collect.label, 'Training Data Stage Distribution');

% interchange dimension, from [epoch, channel, data] to [epoch_i, data, channel]
% to be consistent in Python environment
train_data_collect.data = permute(train_data_collect.data, [1, 3, 2]);

clear('train_i', 'train_path', 'no_train', 'extra', 'i', 'label', 'hdr', ...
      'edf_path','epoch_i', 'labels', 'no_edf', 'no_epochs', 'p', ...
      'psg_raw', 'record', 'temp', 'epoch_iter');
save('train_data_PSG.mat', 'train_data_collect', '-v7.3');
clear('train_data_collect');

%% Read test data
% test data
test_data_collect.data = [];
test_data_collect.label = [];

epoch_iter = 0;

no_test = length(test_path);

for test_i = 1:no_test
    % Read Raw Data
    edf_path = ['psg_data/', test_path{test_i}];
    if ~exist(edf_path, 'file')
        disp(edf_path);
        disp('File not Found. Continue to Next loop...');
        continue;
    end
    [hdr, record] = edfread(edf_path);
    record = record(selected_channels, :);
    % Read labels
    labels = parse_sleep_stage(edf_path);
    no_epochs = length(labels);
    
    test_data_collect.data = [test_data_collect.data; ...
                                zeros(no_epochs-epoch_begin_offset-epoch_end_offset, 4, sample_p_epoch_new)];
    test_data_collect.label = [test_data_collect.label; ...
                                zeros(no_epochs-epoch_begin_offset-epoch_end_offset, num_class)];
                            
    % Use offset to remove beginning few epochs for calibration purposes
    for epoch_i = epoch_begin_offset+1:no_epochs-epoch_end_offset-1
        psg_raw = record(:, ...
            ((epoch_i-1)*sample_p_epoch_ori+1):(epoch_i*sample_p_epoch_ori));
        label = labels(epoch_i, 1);
        
        % remove epochs with "unknown" label
        if ~isnan(label)
            epoch_iter = epoch_iter + 1;
            temp = downsample(psg_raw.', decimation_ratio).';
            test_data_collect.data(epoch_iter, :, :) = raw_data_thresholding(temp, tEEG, tEOG, rEMG);
            test_data_collect.label(epoch_iter, :) = one_hot_with_total_num(label, num_class);
        end
    end
    
    
    % clear extra zeros epochs
    extra = size(test_data_collect.data, 1);
    if extra > epoch_iter
        test_data_collect.data(epoch_iter+1:end, :, :) = [];
        test_data_collect.label(epoch_iter+1:end, :) = [];
    end
end

% Generate histogram of classes to check class imbalance
class_counter(test_data_collect.label, 'Testing Data Stage Distribution');

% interchange dimension, from [epoch, channel, data] to [epoch_i, data, channel]
% to be consistent in Python environment
test_data_collect.data = permute(test_data_collect.data, [1, 3, 2]);

clear('test_i', 'test_path', 'no_test', 'extra', 'i', 'label', 'hdr', ...
      'edf_path','epoch_i', 'labels', 'no_edf', 'no_epochs', 'p', ...
      'psg_raw', 'record', 'temp', 'epoch_iter');
save('test_data_PSG.mat', 'test_data_collect', '-v7.3');
clear('test_data_collect');


%% Read valid data
% valid data
valid_data_collect.data = [];
valid_data_collect.label = [];

epoch_iter = 0;

no_valid = length(valid_path);

for valid_i = 1:no_valid
    % Read Raw Data
    edf_path = ['psg_data/', valid_path{valid_i}];
    if ~exist(edf_path, 'file')
        disp(edf_path);
        disp('File not Found. Continue to Next loop...');
        continue;
    end
    [hdr, record] = edfread(edf_path);
    record = record(selected_channels, :);
    % Read labels
    labels = parse_sleep_stage(edf_path);
    no_epochs = length(labels);
    
    valid_data_collect.data = [valid_data_collect.data; ...
                                zeros(no_epochs-epoch_begin_offset-epoch_end_offset, 4, sample_p_epoch_new)];
    valid_data_collect.label = [valid_data_collect.label; ...
                                zeros(no_epochs-epoch_begin_offset-epoch_end_offset, num_class)];
                            
    % Use offset to remove beginning few epochs for calibration purposes
    for epoch_i = epoch_begin_offset+1:no_epochs-epoch_end_offset-1
        psg_raw = record(:, ...
            ((epoch_i-1)*sample_p_epoch_ori+1):(epoch_i*sample_p_epoch_ori));
        label = labels(epoch_i, 1);
        
        % remove epochs with "unknown" label
        if ~isnan(label)
            epoch_iter = epoch_iter + 1;
            temp = downsample(psg_raw.', decimation_ratio).';
            valid_data_collect.data(epoch_iter, :, :) = raw_data_thresholding(temp, tEEG, tEOG, rEMG);
            valid_data_collect.label(epoch_iter, :) = one_hot_with_total_num(label, num_class);
        end
    end
    
    
    % clear extra zeros epochs
    extra = size(valid_data_collect.data, 1);
    if extra > epoch_iter
        valid_data_collect.data(epoch_iter+1:end, :, :) = [];
        valid_data_collect.label(epoch_iter+1:end, :) = [];
    end
end

% Generate histogram of classes to check class imbalance
class_counter(valid_data_collect.label, 'Validation Data Stage Distribution');

% interchange dimension, from [epoch, channel, data] to [epoch_i, data, channel]
% to be consistent in Python environment
valid_data_collect.data = permute(valid_data_collect.data, [1, 3, 2]);

clear('valid_i', 'valid_path', 'no_valid', 'extra', 'i', 'label', 'hdr', ...
      'edf_path','epoch_i', 'labels', 'no_edf', 'no_epochs', 'p', ...
      'psg_raw', 'record', 'temp', 'epoch_iter');
save('valid_data_PSG.mat', 'valid_data_collect', '-v7.3');
clear('valid_data_collect');
