%%  SGH PSG data preparation
%
% This script is used to prepare SGH PSG data for use in Python environment

clear;
clc;

%% Define Environmental Variables & prepare file path
no_edf = 20;
no_train = 12;
no_test = 4;
no_valid = 4;

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

%% Read data
% Train data
for i = 1:no_train
    [train_edf_data{i, 1}, train_edf_data{i, 2}] = edfread(train_path{i});
end
save('train_edf_data.mat', 'train_edf_data', '-v7.3');
clear('train_edf_data', 'train_path');

% Read test data
for i = 1:no_test
    [test_edf_data{i, 1}, test_edf_data{i, 2}] = edfread(test_path{i});
end
save('test_edf_data.mat', 'test_edf_data', '-v7.3');
clear('test_edf_data', 'test_path');

% Read Valid Data
for i = 1:no_valid
    [valid_edf_data{i, 1}, valid_edf_data{i, 2}] = edfread(valid_path{i});
end
save('valid_edf_data.mat', 'valid_edf_data', '-v7.3');
clear('valid_edf_data', 'valid_path');

