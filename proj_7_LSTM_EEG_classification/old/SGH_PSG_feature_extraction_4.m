%% SGH_EEG_FEATURE_EXTRACTION_4
% This script runs after SGH_EEG_staging_preparation_2
% Tasks performed:
%
% 1. Extract features of different domain from EEG epochs stored in
%    "SGH_EEG_data_reshaped_.mat"
%
% Extract Feature using features defined in EEG_feature_extractor
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

clear;
clc;

%% Train data 1
load('SGH_EEG_data_reshaped_train_1.mat');
train_data_reshaped = train_data_reshaped_1;
clear('train_data_reshaped_1');

% extract features from training data
no_train_data = size(train_data_reshaped, 1);
y_train = zeros(no_train_data, 11);

parfor train_i = 1:no_train_data
    
    x = train_data_reshaped(train_i, :);
    feature_vec = [
        EEG_feature_extractor.extract_relative_power(x, Fs, 'delta'), ...  
        EEG_feature_extractor.extract_relative_power(x, Fs, 'theta'),  ...
        EEG_feature_extractor.extract_relative_power(x, Fs, 'alpha'), ...
        EEG_feature_extractor.extract_relative_power(x, Fs, 'beta'), ...
        EEG_feature_extractor.extract_relative_power(x, Fs, 'gamma'), ...
        EEG_feature_extractor.extract_abs_median(x), ...
        EEG_feature_extractor.extract_kurtosis(x), ...
        EEG_feature_extractor.extract_std(x), ...
        EEG_feature_extractor.extract_entropy(x), ...
        EEG_feature_extractor.extract_spectral_mean(x, Fs), ...
        EEG_feature_extractor.extract_fractal_exponent(x, Fs)  ];
        
    y_train(train_i, :) = EEG_feature_extractor.normalize_feature_vec(feature_vec);
    train_i
end

clear('x', 'feature_vec', 'train_i');

save('SGH_EEG_data_with_feature_train_1.mat', 'Fs', 'mode', ...
     'train_data_reshaped', 'y_train', '-v7.3');
clear('train_data_reshaped', 'y_train');


%% Train data 2
load('SGH_EEG_data_reshaped_train_2.mat');
train_data_reshaped = train_data_reshaped_2;
clear('train_data_reshaped_2');

% extract features from training data
no_train_data = size(train_data_reshaped, 1);
y_train = zeros(no_train_data, 11);

parfor train_i = 1:no_train_data
    
    x = train_data_reshaped(train_i, :);
    feature_vec = [
        EEG_feature_extractor.extract_relative_power(x, Fs, 'delta'), ...  
        EEG_feature_extractor.extract_relative_power(x, Fs, 'theta'),  ...
        EEG_feature_extractor.extract_relative_power(x, Fs, 'alpha'), ...
        EEG_feature_extractor.extract_relative_power(x, Fs, 'beta'), ...
        EEG_feature_extractor.extract_relative_power(x, Fs, 'gamma'), ...
        EEG_feature_extractor.extract_abs_median(x), ...
        EEG_feature_extractor.extract_kurtosis(x), ...
        EEG_feature_extractor.extract_std(x), ...
        EEG_feature_extractor.extract_entropy(x), ...
        EEG_feature_extractor.extract_spectral_mean(x, Fs), ...
        EEG_feature_extractor.extract_fractal_exponent(x, Fs)  ];
        
    y_train(train_i, :) = EEG_feature_extractor.normalize_feature_vec(feature_vec);
    train_i
end

clear('x', 'feature_vec', 'train_i');

save('SGH_EEG_data_with_feature_train_2.mat', 'Fs', 'mode', ...
     'train_data_reshaped', 'y_train', '-v7.3');
clear('train_data_reshaped', 'y_train');

%% Train data 3
load('SGH_EEG_data_reshaped_train_3.mat');
train_data_reshaped = train_data_reshaped_3;
clear('train_data_reshaped_3');

% extract features from training data
no_train_data = size(train_data_reshaped, 1);
y_train = zeros(no_train_data, 11);

parfor train_i = 1:no_train_data
    
    x = train_data_reshaped(train_i, :);
    feature_vec = [
        EEG_feature_extractor.extract_relative_power(x, Fs, 'delta'), ...  
        EEG_feature_extractor.extract_relative_power(x, Fs, 'theta'),  ...
        EEG_feature_extractor.extract_relative_power(x, Fs, 'alpha'), ...
        EEG_feature_extractor.extract_relative_power(x, Fs, 'beta'), ...
        EEG_feature_extractor.extract_relative_power(x, Fs, 'gamma'), ...
        EEG_feature_extractor.extract_abs_median(x), ...
        EEG_feature_extractor.extract_kurtosis(x), ...
        EEG_feature_extractor.extract_std(x), ...
        EEG_feature_extractor.extract_entropy(x), ...
        EEG_feature_extractor.extract_spectral_mean(x, Fs), ...
        EEG_feature_extractor.extract_fractal_exponent(x, Fs)  ];
        
    y_train(train_i, :) = EEG_feature_extractor.normalize_feature_vec(feature_vec);
    train_i
end

clear('x', 'feature_vec', 'train_i');

save('SGH_EEG_data_with_feature_train_3.mat', 'Fs', 'mode', ...
     'train_data_reshaped', 'y_train', '-v7.3');
clear('train_data_reshaped', 'y_train');

mail_notification('ytang014@e.ntu.edu.sg', 'Train Feature Extraction - DONE', ' ');

%% Test data
load('SGH_EEG_data_reshaped_test.mat');

% extract features from testing data
no_test_data = size(test_data_reshaped, 1);
y_test = zeros(no_test_data, 11);

parfor test_i = 1:no_test_data
    
    x = test_data_reshaped(test_i, :);
    feature_vec = [
        EEG_feature_extractor.extract_relative_power(x, Fs, 'delta'), ...  
        EEG_feature_extractor.extract_relative_power(x, Fs, 'theta'),  ...
        EEG_feature_extractor.extract_relative_power(x, Fs, 'alpha'), ...
        EEG_feature_extractor.extract_relative_power(x, Fs, 'beta'), ...
        EEG_feature_extractor.extract_relative_power(x, Fs, 'gamma'), ...
        EEG_feature_extractor.extract_abs_median(x), ...
        EEG_feature_extractor.extract_kurtosis(x), ...
        EEG_feature_extractor.extract_std(x), ...
        EEG_feature_extractor.extract_entropy(x), ...
        EEG_feature_extractor.extract_spectral_mean(x, Fs), ...
        EEG_feature_extractor.extract_fractal_exponent(x, Fs)  ];
        
    y_test(test_i, :) = EEG_feature_extractor.normalize_feature_vec(feature_vec);
    test_i
end

clear('x', 'feature_vec', 'test_i');

save('SGH_EEG_data_with_feature_test.mat', 'Fs', 'mode', ...
     'test_data_reshaped', 'y_test', '-v7.3');
clear('test_data_reshaped', 'y_test');

mail_notification('ytang014@e.ntu.edu.sg', 'Test Feature Extraction - DONE', ' ');

%% valid data
load('SGH_EEG_data_reshaped_valid.mat');

% extract features from validing data
no_valid_data = size(valid_data_reshaped, 1);
y_valid = zeros(no_valid_data, 11);

parfor valid_i = 1:no_valid_data
    
    x = valid_data_reshaped(valid_i, :);
    feature_vec = [
        EEG_feature_extractor.extract_relative_power(x, Fs, 'delta'), ...  
        EEG_feature_extractor.extract_relative_power(x, Fs, 'theta'),  ...
        EEG_feature_extractor.extract_relative_power(x, Fs, 'alpha'), ...
        EEG_feature_extractor.extract_relative_power(x, Fs, 'beta'), ...
        EEG_feature_extractor.extract_relative_power(x, Fs, 'gamma'), ...
        EEG_feature_extractor.extract_abs_median(x), ...
        EEG_feature_extractor.extract_kurtosis(x), ...
        EEG_feature_extractor.extract_std(x), ...
        EEG_feature_extractor.extract_entropy(x), ...
        EEG_feature_extractor.extract_spectral_mean(x, Fs), ...
        EEG_feature_extractor.extract_fractal_exponent(x, Fs)  ];
    
    y_valid(valid_i, :) = EEG_feature_extractor.normalize_feature_vec(feature_vec);
    valid_i
end

clear('x', 'feature_vec', 'valid_i');

save('SGH_EEG_data_with_feature_valid.mat', 'Fs', 'mode', ...
     'valid_data_reshaped', 'y_valid', '-v7.3');
clear('valid_data_reshaped', 'y_valid');

mail_notification('ytang014@e.ntu.edu.sg', 'valid Feature Extraction - DONE', ' ');

