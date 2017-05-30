classdef EEG_signal_processor
%% EEG_SIGNAL_PROCESSOR
% classs containing functions for raw EEG signal processing
% 
% functions contained:
%   
%   threshold_normalize
%   

    properties
        
    end
    
    
    methods (Static)
        
        function data = threshold_normalize(data, threshold, mode)
        % threshold_normalization: 
        % data: raw EEG data
        % threshold: absolute threshold, e.g. 60 means limit signal in +/- 60 
        % mode: 'minmax', 'meanstd', 'range'
            
            % perfrom thresholding
            data(data<-threshold) = -threshold;
            data(data>threshold) = threshold;            
            
            % perform normalziation
            switch(mode)
                case 'minmax'
                    data = (data - min(data)) ./ (max(data) - min(data));
                    
                case 'meanstd'
                    data = (data - mean(data)) ./ std(data);
                    
                case 'range'    % simply scale into 0 - 1 according to threshold
                    data = data ./ (2 * threshold) + 0.5;
                
                otherwise
                    error('Normalization mode not correct');
            end
        end
        
        function data = thresholding(data, threshold)
        % thresholding the data 
        % data: raw EEG data
        % threshold: absolute threshold, e.g. 60 means limit signal in +/- 60 
            
            % perfrom thresholding
            data(data<-threshold) = -threshold;
            data(data>threshold) = threshold;     
        end
        
    end
    
end