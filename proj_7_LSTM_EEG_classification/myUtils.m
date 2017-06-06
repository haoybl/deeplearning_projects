classdef myUtils
    
    methods (Static)
        
        function stage_list = parse_sleep_stage(edf_path)
        % PARSE_SLEEP_STAGE
        % This function is used to extract sleep stage information stored in
        % edf.xml files.
        % The output will be used as classification target.

        %   Note: sleep stage-label number: 
        %   N1	N2	N3	REM	unknown Wake
        %   1	2	3	5	?       0
        %   change to 
        %   N1	N2	N3	REM	unknown Wake
        %   2	3	4	5	NaN     1

            % Define environent variables
            xml_path = [edf_path, '.XML'];
            % Load xml
            xml_data = xml2struct(xml_path);
            stages = xml_data.CMPStudyConfig.SleepStages.SleepStage;
            % Process stages data
            no_epochs = length(stages);
            stage_list = zeros(no_epochs, 1);
            for i = 1:no_epochs
                label = str2double(stages{1, i}.Text);
                if isnan(label) % label = NaN
                    stage_list(i, 1) = nan;
                elseif label < 5  % label = 0, 1, 2, 3
                    stage_list(i, 1) = label + 1;
                else % label = 5
                    stage_list(i, 1) = label;
                end
            end
            fprintf('XML data parsed. In total %d epochs.\n', no_epochs);
        end
        
        function [b, a] = notch_filter_ba(fs, f0)
        % Notch Filter at Given Frequency -- Returns filter coefficients b & a
        % Given Input sequence at sampling frequency Fs, notch filter is designed
        % at notch frequency f0 to remove power line interference.
        % Original Source:
        % http://dsp.stackexchange.com/questions/1088/filtering-50hz-using-a-notch-filter-in-matlab

            fn = fs/2;              %#Nyquist frequency
            freqRatio = f0/fn;      %#ratio of notch freq. to Nyquist freq.

            notchWidth = 0.1;       %#width of the notch

            %Compute zeros
            notchZeros = [exp( sqrt(-1)*pi*freqRatio ), exp( -sqrt(-1)*pi*freqRatio )];

            %#Compute poles
            notchPoles = (1-notchWidth) * notchZeros;

            %figure;
            %zplane(notchZeros.', notchPoles.');

            b = poly( notchZeros ); %# Get moving average filter coefficients
            a = poly( notchPoles ); %# Get autoregressive filter coefficients
   
        end
        
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
        
        function [ output ] = one_hot_with_total_num( label, num_class )
        %ONE_HOT_WITH_TOTAL_NUM Summary of this function goes here
        %   Detailed explanation goes here
            output = zeros(1, num_class);
            output(label) = 1;
        end
        
        function mail_notification(target, subject, msg)
        % EMAIL_NOTIFICATION
        % Generate automatic email notification to target receiver, with subject
        % and message content.
        %
        % This is mainly used to track working process on Matlab servers.

        % Note that the email and password are created solely on purpose of email
        % notification. Any user can use it.

            mail = 'ntutysendmail@gmail.com';   % sender email address
            password = '123456789Aa';           % sender email password
            setpref('Internet','SMTP_Server','smtp.gmail.com');

            setpref('Internet','E_mail',mail);
            setpref('Internet','SMTP_Username',mail);
            setpref('Internet','SMTP_Password',password);
            props = java.lang.System.getProperties;
            props.setProperty('mail.smtp.auth','true');
            props.setProperty('mail.smtp.socketFactory.class', 'javax.net.ssl.SSLSocketFactory');
            props.setProperty('mail.smtp.socketFactory.port','465');

            sendmail(target,subject,msg)
        end
        
        function sample = get_epoch_data(record, epoch, Fs)

            sample = record(((epoch-1)*Fs*30+1) : epoch*Fs*30);

        end
        
        function [ aug_data, aug_label ] = train_data_augmentation( ori_data, ori_label )
        %TRAIN_DATA_AUGMENTATION 
        %   Augment Training Data to allevate class imbalance problem.
        %   The majority of sleep stage is N2, so need to augment the rest
        %   
        %   Current method: 
        %   Check the highest number of class count, randomly sample each
        %   class until every class has reached the number
        %
        %   input:  ori_data:   [no_sample, feature_size, channel]
        %           ori_label:  [no_sample, no_class]
        %   output:
        %         randomly sampled, equally distributed augmented data
            
            % import other functions in this class
            import myUtils.*
        
            % check highest number of class count
            count = class_counter(ori_label);
            max_count = max(count);
            no_class = length(count);
            no_sample = size(ori_data, 1);
            no_feature = size(ori_data, 2);
            no_channel = size(ori_data, 3);
            
            aug_data = zeros(max_count * no_class, no_feature, no_channel);
            aug_label = zeros(max_count * no_class, no_class);
            
            new_count = zeros(1, no_class);
            % Iterate through data many times, until each new_count
            % reaches max_count
            sample_i = 1;
            total_i = 1;
            while total_i <= max_count * no_class 
                
                if sample_i > no_sample
                    sample_i = 1;
                end
                
                data = ori_data(sample_i, :, :);
                label = ori_label(sample_i, :);
                [~, class] = find(label);
                
                % Check the new count of class
                if new_count(class) < max_count
                    aug_data(total_i, :, :) = data;
                    aug_label(total_i, :) = label;
                    new_count(class) = new_count(class) + 1;
                    total_i = total_i + 1;
                end
                sample_i = sample_i + 1;                
            end
            
            perm = randperm(max_count * no_class);
            aug_data = aug_data(perm, :, :);
            aug_label = aug_label(perm, :);

        end
        
        function count = class_counter(one_hot_labels)
        % CLASS_COUNTER
        % Count the number of samples of each class
            count = sum(one_hot_labels);
        end
        
        function class_counter_hist( one_hot_labels, hist_title)
        %CLASS_COUNTER_HIST
        %   Generate histogram of 5 sleep classes

            % Convert one hot labels back to single values
            count = sum(one_hot_labels);

            % Genereate histogram
            figure;
            bar(count);
            title(hist_title);

        end
        
    end
    
end
        