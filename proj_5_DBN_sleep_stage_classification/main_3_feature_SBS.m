%% Introduction
% This is the main file for duplicating the results in
% "Sleep Stage Classification using Unsupervised Feature Learning"
% by Martin Längkvist, Lars Karlsson and Amy Loutfi
%
% This package and the paper can be downloaded from: aass.oru.se\~mlt

%% Initialize
% This section intitializes parameters

clear all

cd('D:\OneDrive\NTU\PhD\PHD DATA\projects\proj_5_DBN_sleep_stage_classification') % set matlab path to installation path\sleep\

%nfiles = 25;

%% Perform Sequential Backwards Search (SBS)

% Load feature matrix
load Eucddb
% for ii=1:nfiles
%     subplot(6,5,ii); plot(E{1}(:,ii)); title(featurenames{ii}); axis tight
% end

nfiles = length(E);

% Pre-process feature matrix
for i = 1:nfiles
    % Make sure E and truestate are the same length
    E{i} = E{i}(1:rows(truestate{i}),:);
    
    % Replace NaN and Inf with 0
    E{i}(isnan(E{i})) = 0;
    E{i}(isinf(E{i})) = 0;
    
    % Normalize some features
    E{i}(:,1:2) = asin(sqrt(E{i}(:,1:2)));              % 'delta, theta EEG'
    E{i}(:,3:5) = log10(E{i}(:,3:5)./(1-E{i}(:,3:5)));  % 'alpha, beta, high EEG'
    E{i}(:,16)=E{i}(:,16)./median(E{i}(:,16));          % 'median EMG'
    E{i}(:,18)=log10(1+E{i}(:,18));                     % 'kurtosis EEG'
    E{i}(:,19)=log10(1+E{i}(:,19));                     % 'kurtosis EOG 1'
    E{i}(:,20)=log10(1+E{i}(:,20));                     % 'kurtosis EMG'
    E{i}(:,21)=log10(1+E{i}(:,21));                     % 'std EOG 1'
    E{i}(:,22)=log10(1+E{i}(:,22));                     % 'entropy EEG'
    E{i}(:,23)=log10(1+E{i}(:,23));                     % 'entropy EOG 1'
    E{i}(:,24)=log10(1+E{i}(:,24));                     % 'entropy EMG'
    
    % Subtract mean and divide by standard division
    E{i} = zscore(E{i});
end

% Parameters for SBS
NDimensions = 5;  % number of PCA components
NComponents = 5;  % number of GMM components
nstate = 5;       % number of classes
maxsamples = 1000; % maximum GMM training examples (10000 is used in article)
pseudoA = 0.01*ones(5, 5);
pseudoB = 0.01*ones(5, NComponents);

file = struct();

for testingset = 1:nfiles
    % Reset feature vectors
    defaultfeat = []; % Select features that should be guaranteed to be selected in the SBS-algorithm
    feat = 1:28;      % Features to chose from in the SBS-algorithm
    
    % Divide into training, validation, and test sets
    % iterate each file as test set, split the rest, half as training half
    % as validation
    [trainingsets validationsets] = split(removerows((1:nfiles)', testingset)', [0.5 0.5]); %50-50 between train and test
    
    %Reduced Training set
    % The following data is removed:
    %  - 30 second before and after a sleep stage change         ####
    %  - Rows with NaN or Inf values
    %  - Data with annotated artifacts or indetermined are
    %  - Any duplicate rows
    %  - Random data from each sleep stage until class balance   ####
    %  - Further remove data to cap at maxsamples
    % Data is sorted
    E_train = [];
    truestate_train = [];
    for i = trainingsets
        SwitchIndex = min(abs([diff(downsample(truestate{i}, 30)); 0]),1);
        SwitchIndex = min(SwitchIndex + circshift(SwitchIndex,1),1);
        SwitchIndex = myupsample(SwitchIndex,30);
        NanInfIndex = sum(isnan(E{i})+isinf(E{i}),2)>0;
        Stage0Index = truestate{i}==0;
        removeIndex=SwitchIndex+NanInfIndex+Stage0Index;
        E_temp = removerows(E{i}, find(removeIndex>0));
        truestate_temp = removerows(double(truestate{i}), find(removeIndex>0));
        E_train = [E_train; E_temp];
        truestate_train = [truestate_train; truestate_temp];
    end
    [E_train I]=unique(E_train, 'rows');
    truestate_train=truestate_train(I);
    selectedSamples = [];
    samplesPerStage = min(hist(truestate_train,1:5));
    for i=1:5
        a = find(truestate_train==i);
        k = randperm(length(a));
        selectedSamples = [selectedSamples; a(k(1:samplesPerStage))];
    end
    E_train = E_train(selectedSamples,:);
    truestate_train = truestate_train(selectedSamples);
    if length(truestate_train)>maxsamples
        k = randperm(length(truestate_train));
        E_train=E_train(k(1:maxsamples),:);
        truestate_train=truestate_train(k(1:maxsamples));
    end
    
    % Training HMM set
    % The following data is removed:
    %  - Rows with NaN or Inf values
    %  - Data with annotated artifacts or indetermined are
    E_trainHMM = [];
    truestate_trainHMM = [];
    for i = trainingsets
        NanInfIndex = sum(isnan(E{i})+isinf(E{i}),2)>0;
        Stage0Index = truestate{i}==0;
        removeIndex=NanInfIndex+Stage0Index;
        E_temp = removerows(E{i}, find(removeIndex>0));
        truestate_temp = removerows(double(truestate{i}), find(removeIndex>0));
        E_trainHMM = [E_trainHMM; E_temp];
        truestate_trainHMM = [truestate_trainHMM; truestate_temp];
    end
    
    % Validation set
    % The following data is removed:
    %  - Rows with NaN or Inf values
    %  - Data with annotated artifacts or indetermined are
    E_val = [];
    truestate_val = [];
    for i = validationsets
        NanInfIndex = sum(isnan(E{i})+isinf(E{i}),2)>0;
        Stage0Index = truestate{i}==0;
        removeIndex=NanInfIndex+Stage0Index;
        E_temp = removerows(E{i}, find(removeIndex>0));
        truestate_temp = removerows(double(truestate{i}), find(removeIndex>0));
        E_val = [E_val; E_temp];
        truestate_val = [truestate_val; truestate_temp];
    end
    
    % SBS algorithm
    for resultrow = length(feat)+length(defaultfeat):-1:max(NDimensions,length(defaultfeat))+1
        validation=zeros(1,length(feat));
        for resultcol = 1:length(feat)
            fprintf('File %i of %i. Removing feature %i of %i\n', testingset, nfiles, resultcol, length(feat));
            
            % Remove one feature from subset of features
            SBSfeat=[removerows(feat',resultcol)' defaultfeat];
            
            % Train PCA and GMM with reduced training set
            coefs = princomp(E_train(:,SBSfeat));
            E_trainPCA = E_train(:,SBSfeat)*coefs(:,1:NDimensions);
            obj = gmdistribution.fit(E_trainPCA, NComponents, 'Start', truestate_train);
            
            % Train HMM with not reduced training set
            [outcome nlog P] = cluster(obj, E_trainHMM(:,SBSfeat)*coefs(:,1:NDimensions));
            [A B] = hmmestimate(outcome, truestate_trainHMM, 'Pseudotransitions', pseudoA , 'Pseudoemissions', pseudoB);
            
            % Validate selected features using classification accuracy from validation set
            outcome = cluster(obj, E_val(:,SBSfeat)*coefs(:,1:NDimensions));
            state = hmmviterbi(outcome, A, B);
            acc = PlotResult(state, truestate_val, false);
            
            % Store result
            validation(resultcol) = acc;
        end
        
        % Remove feature that gave best result
        [val, row] = max(validation);
        removedfeat = feat(row);
        feat = removerows(feat',row)';
        
        % Store results
        file(testingset).iteration(resultrow).val=val;
        file(testingset).iteration(resultrow).removedfeat = removedfeat;
        file(testingset).iteration(resultrow).feat = feat;
    end
end

save('SBSresults', 'file', 'E');

% Plot results
for i=1:nfiles
    figure;
    accuracy = [file(i).iteration(28:-1:5).val];
    plot(accuracy, 'x-')
    removedfeatvector = [file(i).iteration(28:-1:5).removedfeat];
    set(gca, 'YMinorGrid','on', 'position', [0.13 0.25 0.775 0.65], 'XTick', 1:length(removedfeatvector), 'XTickLabel', featurenames(removedfeatvector));
    axis tight; axis([1 length(removedfeatvector) 0.6 0.8])
    rotateticklabel(gca,90); ylabel('Classification accuracy [%]');
end

