function stage_list = parse_sleep_stage(edf_path)
%% PARSE_SLEEP_STAGE
% This function is used to extract sleep stage information stored in
% edf.xml files.
% The output will be used as classification target.

%   Note: sleep stage-label number: 
%   N1	N2	N3	REM	unknown Wake
%   1	2	3	5	?       0

%% Define environent variables

xml_path = [edf_path, '.XML'];

%% Load xml
xml_data = xml2struct(xml_path);
stages = xml_data.CMPStudyConfig.SleepStages.SleepStage;

%% Process stages data
no_epochs = length(stages);
stage_list = zeros(no_epochs, 1);

for i = 1:no_epochs
    stage_list(i, 1) = str2double(stages{1, i}.Text);
end

fprintf('XML data parsed. In total %d epochs.\n', no_epochs);

end