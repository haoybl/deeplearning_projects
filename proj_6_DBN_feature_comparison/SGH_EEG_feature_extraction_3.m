%% SGH_EEG_FEATURE_EXTRACTION_3
% This script runs after SGH_EEG_staging_preparation_2
% Tasks performed:
%
% 1. Reshape EEG data into 1-sec windows out of 30-sec epochs
% 
% 2. Thresholding (and Normalizing) raw EEG data before feature extraction
%
% 3. Extract features of different domain from EEG epochs stored in
%    "SGH_EEG_data_prepared.mat"
%
% 4. For each feature, prepare training, testing and validation set

clear;
clc;

%% Load data & define meta variables

load('SGH_EEG_data_prepared.mat');
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

clear('train_data_collect', 'train_size', 'no_train_reshaped', 'count', ...
      'sample_i', 'channel_j','temp', 'temp_size');

% reshape test data
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

clear('test_data_collect', 'test_size', 'no_test_reshaped', 'count', ...
      'sample_i', 'channel_j','temp', 'temp_size');

% reshape validation data
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

clear('valid_data_collect', 'valid_size', 'no_valid_reshaped', 'count', ...
      'sample_i', 'channel_j','temp', 'temp_size');

%% Extract Feature using features defined in EEG_feature_extractor
% Feature used: 
% note: index is consistent with EEG_feature_extractor class
%
%    1. relative power in frequency band, delta, theta, alpha, beta and gamma 
%    2. absolute median
%    4. kurtosis 
%    5. std
%    6. entropy
%    7  spectral mean
%    8. fractal exponent

% extract features from training data
no_train_data = size(train_data_reshaped, 1);
y_train = zeros(no_train_data, 11);

parfor train_i = 1:no_train_data
    
    x = train_data_reshaped(train_i, :);
    feature_vec = [
        EEG_feature_extractor.extract_relative_power(x, Fs, [0.5, 4]), ...  
        EEG_feature_extractor.extract_relative_power(x, Fs, [4, 8]),  ...
        EEG_feature_extractor.extract_relative_power(x, Fs, [8, 13]), ...
        EEG_feature_extractor.extract_relative_power(x, Fs, [13, 20]), ...
        EEG_feature_extractor.extract_relative_power(x, Fs, [20, Fs/2]), ...
        EEG_feature_extractor.extract_abs_median(x), ...
        EEG_feature_extractor.extract_kurtosis(x), ...
        EEG_feature_extractor.extract_std(x), ...
        EEG_feature_extractor.extract_entropy(x), ...
        EEG_feature_extractor.extract_spectral_mean(x, Fs), ...
        EEG_feature_extractor.extract_fractal_exponent(x, Fs)   ];
        
    y_train(train_i, :) = feature_vec;
    train_i
end

clear('x', 'feature_vec', 'train_i');

% extract features from test data
no_test_data = size(test_data_reshaped, 1);
y_test = zeros(no_test_data, 8);

parfor test_i = 1:no_test_data
    
    x = test_data_reshaped(test_i, :);
    feature_vec = [
        EEG_feature_extractor.extract_relative_power(x, Fs, [0.5, 4]), ...  
        EEG_feature_extractor.extract_relative_power(x, Fs, [4, 8]),  ...
        EEG_feature_extractor.extract_relative_power(x, Fs, [8, 13]), ...
        EEG_feature_extractor.extract_relative_power(x, Fs, [13, 20]), ...
        EEG_feature_extractor.extract_relative_power(x, Fs, [20, Fs/2]), ...
        EEG_feature_extractor.extract_abs_median(x), ...
        EEG_feature_extractor.extract_kurtosis(x), ...
        EEG_feature_extractor.extract_std(x), ...
        EEG_feature_extractor.extract_entropy(x), ...
        EEG_feature_extractor.extract_spectral_mean(x, Fs), ...
        EEG_feature_extractor.extract_fractal_exponent(x, Fs)   ];
        
    y_test(test_i, :) = feature_vec;
end

clear('x', 'feature_vec', 'test_i');

% extract features from validation data
no_valid_data = size(valid_data_reshaped, 1);
y_valid = zeros(no_valid_data, 8);

parfor valid_i = 1:no_valid_data
    
    x = valid_data_reshaped(valid_i, :);
    feature_vec = [
        EEG_feature_extractor.extract_relative_power(x, Fs, [0.5, 4]), ...  
        EEG_feature_extractor.extract_relative_power(x, Fs, [4, 8]),  ...
        EEG_feature_extractor.extract_relative_power(x, Fs, [8, 13]), ...
        EEG_feature_extractor.extract_relative_power(x, Fs, [13, 20]), ...
        EEG_feature_extractor.extract_relative_power(x, Fs, [20, Fs/2]), ...
        EEG_feature_extractor.extract_abs_median(x), ...
        EEG_feature_extractor.extract_kurtosis(x), ...
        EEG_feature_extractor.extract_std(x), ...
        EEG_feature_extractor.extract_entropy(x), ...
        EEG_feature_extractor.extract_spectral_mean(x, Fs), ...
        EEG_feature_extractor.extract_fractal_exponent(x, Fs)   ];
        
    y_valid(valid_i, :) = feature_vec;
end

clear('x', 'feature_vec', 'valid_i');

save('SGH_EEG_data_and_feature.mat', 'train_data_reshaped', 'test_data_reshaped', ...
     'valid_data_reshaped', 'mode', 'threshold', 'y_train', 'y_test', 'y_valid');


