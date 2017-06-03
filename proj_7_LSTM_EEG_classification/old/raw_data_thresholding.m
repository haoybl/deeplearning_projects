function data = raw_data_thresholding(data, tEEG, tEOG, rEMG)
    
    % data in form [channel, raw]
    % channels are EEG, EEG, EOG, EMG
    % note EMG is multiplied by ratio rEMG
    % normalization is performed by dividing threshold value
    % final result: all EEG, EOG and EMG are scaled into [-1, 1] range
    
    EEG_data = data(1:2, :);
    EEG_data(EEG_data>tEEG) = tEEG;
    EEG_data(EEG_data<-tEEG) = -tEEG;
    EEG_data = EEG_data / tEEG;
    
    EOG_data = data(3, :);
    EOG_data(EOG_data>tEOG) = tEOG;
    EOG_data(EOG_data<-tEOG) = -tEOG;
    EOG_data = EOG_data / tEOG;
   
    EMG_data = data(4, :) * rEMG;
    
    data = [EEG_data; EOG_data; EMG_data];
end