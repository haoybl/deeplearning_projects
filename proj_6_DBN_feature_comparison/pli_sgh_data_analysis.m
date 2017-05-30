%% PLI_SGH_DATA_ANALYSIS
% This script is mainly used to examine raw PSG data collected from SGH.
% Regarding to powerline interference and prefilter.

clear;
clc;

% load raw data
load('test_edf_data.mat');


%% define environmental variables

EEG_fs = 256;
EMG_fs = 256;

subject = [2, 3, 4];
epoch = [300, 500, 700];

%% Plot fft of raw data
% EEG Data
figure;
plot_i = 1;
for sbj_i = 1:3
    for epoch_i = 1:3
        
        sample = get_epoch_data(test_edf_data{subject(sbj_i), 2}(1, :), ...
                                epoch(epoch_i), EEG_fs);
        sample = (sample - mean(sample))/std(sample);
        title_str = ['raw EEG - sbj: ', num2str(subject(sbj_i)), ...
            ' epoch: ', num2str(epoch(epoch_i))];
        
        subplot(3, 3, plot_i);
        fft_test(sample, EEG_fs, title_str);
        
        ylim([0, 0.3])
        
        plot_i = plot_i + 1;
    end
end

% EMG data
figure;
plot_i = 1;
for sbj_i = 1:3
    for epoch_i = 1:3
        
        sample = get_epoch_data(test_edf_data{subject(sbj_i), 2}(7, :), ...
                                epoch(epoch_i), EMG_fs);
        sample = (sample - mean(sample))/std(sample);
        title_str = ['raw EMG - sbj: ', num2str(subject(sbj_i)), ...
            ' epoch: ', num2str(epoch(epoch_i))];

        subplot(3, 3, plot_i);
        fft_test(sample, EMG_fs, title_str);
        ylim([0, 0.15]);
        plot_i = plot_i + 1;
    end
end

