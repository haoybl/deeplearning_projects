%% DBN on features (feat-DBN)
clear all;
nfiles = 5;
addpath('.\DBNtoolbox\code') %Code for DBN

% Load feature matrix
load('Eucddb.mat')

% Normalize features
for i = 1:nfiles
    % Make E and truestate the same length
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
    
    % Divide by (max(feature)-min(feature)) and adding 0.5 to keep values around [0 1]
    E{i}=bsxfun(@rdivide, E{i}, (max(E{i})-min(E{i}))) + 0.5;
end

results=struct([]);

for testingsets=1:nfiles
    %Divide into training and testing set
    trainingsets = removerows((1:nfiles)',testingsets)';
    data=[];
    labels=[];
    testdata=[];
    testlabels=[];
    for i=trainingsets
        SwitchIndex = min(abs([diff(downsample(truestate{i},30)); 0]),1);
        SwitchIndex = min(SwitchIndex + circshift(SwitchIndex,1),1); %remove 30s before and after switch
        SwitchIndex = myupsample(SwitchIndex,30);
        E{i}=E{i}(SwitchIndex~=1,:);
        truestate{i}=truestate{i}(SwitchIndex~=1,:);
        
        data=[data; E{i}];
        labels=[labels; truestate{i}];
    end
    for i=testingsets
        testdata=[testdata; E{i}];
        testlabels=[testlabels; truestate{i}];
    end
    
    % Balance samples based on category prevalence
    rand('state',0);
    samplesPerClass=min([sum(labels==1) sum(labels==2) sum(labels==3) sum(labels==4) sum(labels==5)]);
    newdata=[];
    newlabel=[];
    for i=1:5
        row = find(labels==i);
        selectedSamples=randperm(length(row));
        newlabel=[newlabel; labels(row(selectedSamples(1:samplesPerClass)))];
        newdata=[newdata; data(row(selectedSamples(1:samplesPerClass)),:)];
    end
    data=newdata;
    labels=newlabel;
    clear newlabel newdata selectedSamples samplesPerClass
    
    % Divide training set into train and validation subsets
    rand('state',0);
    k=randperm(length(data));
    %k=randperm(250000);
    traindata=data(k(1:floor(length(data)*5/6)),:);
    valdata=data(k(floor(length(data)*5/6)+1:end),:);
    trainlabels=labels(k(1:floor(length(data)*5/6)));
    vallabels=labels(k(floor(length(data)*5/6)+1:end));
    fprintf('Train \t Val\n');
    for labeliter=1:5;
        fprintf('%i \t %i\n', sum(trainlabels==labeliter), sum(vallabels==labeliter));
    end
    
    % ---------------------------- Train --------------------------------------
    layerSize = [50 50]; %200 200 is used in article
    
    % Parameters
    rbmParams.numEpochs = 20; %100 is used in article
    rbmParams.verbosity = 1;
    rbmParams.miniBatchSize = 1000;
    rbmParams.attemptLoad = 0;
    dbnParams.numEpochs = 10; % 50 is used in article
    dbnParams.verbosity = 1;
    dbnParams.miniBatchSize = 1000;
    dbnParams.attemptLoad = 0;
    
    % remove any previously trained model .mat files, otherwise it will load it and continue training
    dnntic = tic;
    nnLayers = GreedyLayerTrain(traindata, valdata, layerSize, 'RBM', rbmParams);
    dnnObj = DeepNN(nnLayers, dbnParams);
    fprintf('Unsupervised backprop...\n');
    dnnObj.Train(traindata, valdata);
    fprintf('Supervised backprop...\n');
    dnnObj.Train(traindata, valdata, trainlabels, vallabels);
    fprintf('Finished training DeepNN.\n\n');
    simtime = toc(dnntic);
    
    %Inference on train data
    [topActivs, ~] = dnnObj.PropLayerActivs(data);
    [~,ytrain] = max(topActivs,[],2);
    ytrain=single(ytrain);
    
    %HMM on inference results
    [A, B] = hmmestimate(ytrain, labels, 'PSEUDOTRANSITIONS', 0.001*ones(5, 5), 'PSEUDOEMISSIONS',0.001*ones(5, 5));
    
    %Inference on test data
    [topActivs2, ~] = dnnObj.PropLayerActivs(testdata);
    [~, ytest] = max(topActivs2,[],2);
    ytest=single(ytest);
    
    ytestHMM = hmmviterbi(ytest, A, B)';
    
    figure;
    subplot(2,1,1); acc=PlotResult(ytest,testlabels);
    subplot(2,1,2); accHMM=PlotResult(ytestHMM,testlabels);
    drawnow
    
    results(testingsets).acc=acc;
    results(testingsets).accHMM=accHMM;
    results(testingsets).ytest=ytest;
    results(testingsets).ytestHMM=ytestHMM;
    results(testingsets).A=A;
    results(testingsets).B=B;
    results(testingsets).simtime=simtime;
    results(testingsets).nnLayers=nnLayers;
    results(testingsets).dnnObj=dnnObj;
    results(testingsets).testlabels=testlabels;
    
    save resultsFEAT results
end