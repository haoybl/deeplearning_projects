%% Calculate classification accuracy on testing sets (feat-GOHMM)
clear all;
clc;
load SBSresults
load Eucddb
maxsamples = 10000;
NDimensions = 5;  % number of PCA components
NComponents = 5;  % number of GMM components
pseudoA = 0.01*ones(5, 5);
pseudoB = 0.01*ones(5, NComponents);

for testingset = 1:nfiles
    % Divide into training and test sets
    trainingsets = removerows((1:nfiles)', testingset)';
    
    %Reduced Training set
    % The following data is removed:
    %  - 30 second before and after a sleep stage change
    %  - Rows with NaN or Inf values
    %  - Data with annotated artifacts or indetermined are
    %  - Any duplicate rows
    %  - Random data from each sleep stage until class balance
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
    
    % Testing set
    % The following data is removed:
    %  - Rows with NaN or Inf values
    %  - Data with annotated artifacts or indetermined are
    E_test = [];
    truestate_test = [];
    for i = testingset
        NanInfIndex = sum(isnan(E{i})+isinf(E{i}),2)>0;
        Stage0Index = truestate{i}==0;
        removeIndex=NanInfIndex+Stage0Index;
        E_temp = removerows(E{i}, find(removeIndex>0));
        truestate_temp = removerows(double(truestate{i}), find(removeIndex>0));
        E_test = [E_test; E_temp];
        truestate_test = [truestate_test; truestate_temp];
    end
    
    % Select best feature subset
    SBSfeat=file(testingset).iteration(5+argmax([file(testingset).iteration(:).val])).feat;
    
    % Train PCA and GMM with reduced training set
    coefs = princomp(E_train(:,SBSfeat));
    E_trainPCA = E_train(:,SBSfeat)*coefs(:,1:NDimensions);
    obj = gmdistribution.fit(E_trainPCA, NComponents, 'Start', truestate_train); %prevents ill-conditioned covariance matrix
    
    % Train HMM with not reduced training set
    [outcome nlog P] = cluster(obj, E_trainHMM(:,SBSfeat)*coefs(:,1:NDimensions));
    [A B] = hmmestimate(outcome, truestate_trainHMM, 'Pseudotransitions', pseudoA , 'Pseudoemissions', pseudoB);
    
    % Validate selected features using classification accuracy from validation set
    outcome = cluster(obj, E_test(:,SBSfeat)*coefs(:,1:NDimensions));
    state = hmmviterbi(outcome, A, B);
    figure(testingset)
    acc = PlotResult(state, truestate_test, true);
end
