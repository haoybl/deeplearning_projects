%% SGH_EEG_RESHAPE_TRAIN_3
%   ----------------------------------------------
%   NOTE : Only for training data to save memory 
%   ----------------------------------------------
% This script runs after SGH_EEG_staging_preparation_2
% Tasks performed:
%
% 1. Reshape EEG data into 1-sec windows out of 30-sec epochs
% 
% 2. Thresholding (and Normalizing) raw EEG data before feature extraction

clear;
clc;

%% Load data & define meta variables

load('SGH_EEG_data_prepared.mat', 'train_data_collect', 'decimation_freq');
Fs = decimation_freq;
clear('decimation_freq');

% EEG raw signal threshold
% +/- 60 uV  according Martin's paper (unsupervised feature... 2012) 
threshold = 60;
% EEG raw signal normalization
% range normalization, simply scale range into 0 - 1 by dividing 2 times
% threhold and plus 0.5
mode = 'range';


%% reshape EEG data to 1-sec window & threshold_normalize

% reshape training data
train_size = size(train_data_collect.data);
no_train_reshaped = prod(train_size)/Fs;

train_data_reshaped = zeros(no_train_reshaped, Fs);

count = 1; 
for sample_i = 1:train_size(1)
    for channel_j = 1:train_size(2)
        temp = squeeze(train_data_collect.data(sample_i, channel_j, :));
        temp = EEG_signal_processor.threshold_normalize(temp, threshold, mode);
        temp = vec2mat(temp, Fs);
        temp_size = size(temp, 1); % default size(temp) ==> 30, 128
        
        train_data_reshaped(temp_size*(count-1)+1:temp_size*count, :) = temp;
        count = count + 1;
    end
end

save('SGH_EEG_data_reshaped_train.mat', 'train_data_reshaped', 'Fs', 'threshold', 'mode');

clear('train_data_collect', 'train_size', 'no_train_reshaped', 'count', ...
      'sample_i', 'channel_j','temp', 'temp_size', 'train_data_reshaped');

% reshape test data
load('SGH_EEG_data_prepared.mat', 'test_data_collect');

test_size = size(test_data_collect.data);
no_test_reshaped = prod(test_size)/Fs;

test_data_reshaped = zeros(no_test_reshaped, Fs);

count = 1; 
for sample_i = 1:test_size(1)
    for channel_j = 1:test_size(2)
        temp = squeeze(test_data_collect.data(sample_i, channel_j, :));
        temp = EEG_signal_processor.threshold_normalize(temp, threshold, mode);
        temp = vec2mat(temp, Fs);
        temp_size = size(temp, 1); % default size(temp) ==> 30, 128
        
        test_data_reshaped(temp_size*(count-1)+1:temp_size*count, :) = temp;
        count = count + 1;
    end
end

save('SGH_EEG_data_reshaped_test.mat', 'test_data_reshaped', 'Fs', 'threshold', 'mode');

clear('test_data_collect', 'test_size', 'no_test_reshaped', 'count', ...
      'sample_i', 'channel_j','temp', 'temp_size', 'test_data_reshaped');

  
% reshape valid data
load('SGH_EEG_data_prepared.mat', 'valid_data_collect');

valid_size = size(valid_data_collect.data);
no_valid_reshaped = prod(valid_size)/Fs;

valid_data_reshaped = zeros(no_valid_reshaped, Fs);

count = 1; 
for sample_i = 1:valid_size(1)
    for channel_j = 1:valid_size(2)
        temp = squeeze(valid_data_collect.data(sample_i, channel_j, :));
        temp = EEG_signal_processor.threshold_normalize(temp, threshold, mode);
        temp = vec2mat(temp, Fs);
        temp_size = size(temp, 1); % default size(temp) ==> 30, 128
        
        valid_data_reshaped(temp_size*(count-1)+1:temp_size*count, :) = temp;
        count = count + 1;
    end
end

save('SGH_EEG_data_reshaped_valid.mat', 'valid_data_reshaped', 'Fs', 'threshold', 'mode');

clear('valid_data_collect', 'valid_size', 'no_valid_reshaped', 'count', ...
      'sample_i', 'channel_j','temp', 'temp_size', 'valid_data_reshaped');




