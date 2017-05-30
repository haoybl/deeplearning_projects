classdef EEG_feature_extractor
%% EEG_FEATURE_EXTRACTOR
% This class defines a collection of methods for extracting features of
% different domain from EEG signals.
%
% Features are listed below:
% 
%    1. relative power in frequency band, delta, theta, alpha, beta and gamma 
%    2. absolute median
%    3. cross-correlation (of two signals)
%    4. kurtosis 
%    5. std
%    6. entropy, 
%    7  spectral mean
%    8. fractal exponent
     
    properties
        epoch_length = 30;
        sampling_frequency = 128;
    end

    methods (Static)
        
        function y = extract_relative_power(x, Fs, powerband)
            % RELATIVE average power of frequency band
            y_delta = bandpower(x, Fs, [0.5, 4]);
            y_theta = bandpower(x, Fs, [4, 8]);
            y_alpha = bandpower(x, Fs, [8, 13]);
            y_beta = bandpower(x, Fs, [13, 20]);
            y_gamma = bandpower(x, Fs, [20, Fs/2]);
            
            y_total = y_delta + y_theta + y_alpha + y_beta + y_gamma;
            
            switch powerband
                case 'delta'
                    y = y_delta / y_total;
                case 'theta'
                    y = y_theta / y_total;
                case 'alpha'
                    y = y_alpha / y_total;
                case 'beta'
                    y = y_beta / y_total;
                case 'gamma'
                    y = y_gamma / y_total;
                otherwise
                    error('Invalid Frequency Band');
            end
        end
        
        function y = extract_abs_median(x)
            y = median(abs(x));
        end
        
        function y = extract_eye_correlation(x1, x2)
            % eye correlation / cross-correlation
            y = xcorr(x1,x2,0,'coef');
        end
        
        function y = extract_kurtosis(x)
            % kurtosis
            y = kurtosis(x);
        end
        
        function y = extract_std(x)
            % standard deviation
            y = std(x);
        end
        
        function y = extract_entropy(x)
            % shannon entropy
            n = length(x);
            P = hist(x,ceil(sqrt(n)))/n;
            y = -nansum(P.*log(P));
        end
        
        function y = extract_spectral_mean(x, Fs)
            % spectral mean
            % weighted average of spectral bands
            % freq_bands = [0.5, 4; 4, 8; 8, 13; 13, 20; 20, Fs/2];
            freq_bands = {'delta'; 'theta'; 'alpha'; 'beta'; 'gamma'};
            spectral_dist = [3.5; 4; 5; 7; Fs/2-20];
            
            for band_i = 1:size(freq_bands, 1)
                spectral_dist(band_i, 2) = ...
                   EEG_feature_extractor.extract_relative_power(x, Fs, freq_bands{band_i, 1});
            end
            
            y = sum(spectral_dist(:, 1).*spectral_dist(:, 2))/(Fs/2-0.5);
        end
        
        function y = extract_fractal_exponent(x, Fs)
            % fractal Exponent: negative slope of linear fit of spectral
            % density in the double logarithmic graph
            % refer to paper
            % res: frequency resolution
            % code copied from Martin's original Paper (unsupervised feature, 2012)
            
            x = x';
            
            T=1/Fs;
            res = 20;   
            freq_bins = length(x)*res;

            Y = T/freq_bins*abs(fft([x; zeros((res-1)*length(x),1)])).^2;
            f = (0:freq_bins-1)/freq_bins/T;
            
            f=f(f>=0.5 & f<Fs/2);
            Y=Y(f>=0.5 & f<Fs/2);
            
            P=polyfit(log(f'),log(Y),1);
            y = P(1);
        end
        
        function feature_vec = normalize_feature_vec(feature_vec)
            % Normalize features according to Martin's paper (unsupervised..., 2012)
            
            feature_vec(1:2) = asin(sqrt(feature_vec(1:2)));                    % 'delta, theta EEG'
            feature_vec(3:5) = log10(feature_vec(3:5)./(1-feature_vec(3:5)));   % 'alpha, beta, high EEG'
            feature_vec(7)   = log10(1+feature_vec(7));         % kurtosis
            feature_vec(8)   = log10(1+feature_vec(8));         % std
            feature_vec(9)   = log10(1+feature_vec(9));         % entropy
        end
            
    end
    
end