function stage_list = parse_sleep_stage(edf_path)
%% PARSE_SLEEP_STAGE
% This function is used to extract sleep stage information stored in
% edf.xml files.
% The output will be used as classification target.

%   Note: sleep stage-label number: 
%   N1	N2	N3	REM	unknown Wake
%   1	2	3	5	?       0
%
%  change to 
%   N1	N2	N3	REM	unknown Wake
%   2	3	4	5	NaN     1

%% Define environent variables

xml_path = [edf_path, '.XML'];

%% Load xml
xml_data = xml2struct(xml_path);
stages = xml_data.CMPStudyConfig.SleepStages.SleepStage;

%% Process stages data
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