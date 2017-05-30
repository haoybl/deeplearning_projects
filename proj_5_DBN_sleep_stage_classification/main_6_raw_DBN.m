%% DBN on raw data (raw-DBN)
clear all;
nfiles = 5;
filek = {'02' '03' '05' '06' '07'};
addpath('.\DBNtoolbox\code\');

% Create visible layer
C=cell(nfiles,1);
Clabel=cell(nfiles,1);
for i=1:nfiles
    %Raw data preprocessing
    cutoff=[0.3 0.4 0.4 0.5]; %raw data not scaled to Hz yet.
    load(['.\data\p' filek{i} '.mat']);
    h = HDR.SampleRate;
    EEG=s(:,1);
    EOG1=s(:,2);
    EOG2=s(:,3);
    EMG=s(:,4);
    
    % Cut-off signals
    EEG(EEG<-cutoff(1))=-cutoff(1);
    EEG(EEG>cutoff(1))=cutoff(1);
    EOG1(EOG1<-cutoff(2))=-cutoff(2);
    EOG1(EOG1>cutoff(2))=cutoff(2);
    EOG2(EOG2<-cutoff(3))=-cutoff(3);
    EOG2(EOG2>cutoff(3))=cutoff(3);
    EMG(EMG<-cutoff(4))=-cutoff(4);
    EMG(EMG>cutoff(4))=cutoff(4);
    
    % Normalize to [0 1] values
    EEG=EEG/(cutoff(1)*2)+0.5;
    EOG1=EOG1/(cutoff(2)*2)+0.5;
    EOG2=EOG2/(cutoff(3)*2)+0.5;
    EMG=EMG/(cutoff(4)*2)+0.5;
    
    % Segment into 1 second windows
    EEG = reshape(EEG,h,length(EEG)/h)';
    EOG1 = reshape(EOG1,h,length(EOG1)/h)';
    EOG2 = reshape(EOG2,h,length(EOG2)/h)';
    EMG = reshape(EMG,h,length(EMG)/h)';
    Clabel{i}=downsample(v,h);
    C{i}=[EEG EOG1 EOG2 EMG]; %concatenate signals
    clear s v HDR EEG EOG1 EOG2 EMG cutoff
end

results=struct([]);
for testingsets=1:nfiles
    % Divide into training and testing sets
    trainingsets = removerows((1:nfiles)',testingsets)';
    data=[];
    labels=[];
    testdata=[];
    testlabels=[];
    for i=trainingsets
        SwitchIndex = min(abs([diff(downsample(Clabel{i},30)); 0]),1);
        SwitchIndex = min(SwitchIndex + circshift(SwitchIndex,1),1);
        SwitchIndex = myupsample(SwitchIndex,30);
        C{i}=C{i}(SwitchIndex~=1,:);
        Clabel{i}=Clabel{i}(SwitchIndex~=1,:);
        data=[data; C{i}];
        labels=[labels; Clabel{i}];
    end
    for i=testingsets
        testdata=[testdata; C{i}];
        testlabels=[testlabels; Clabel{i}];
    end
    clear C Clabel
    
    % Balance samples based on category prevalence
    rand('state',0);
    samplesPerClass=min([sum(labels==1) sum(labels==2) sum(labels==3) sum(labels==4) sum(labels==5)]);
    templabels=[];
    tempdata=[];
    for i=1:5
        row = find(labels==i);
        selectedSamples=randperm(length(row));
        tempdata=[tempdata; data(row(selectedSamples(1:samplesPerClass)),:)];
        templabels=[templabels; labels(row(selectedSamples(1:samplesPerClass)))];
    end
    clear selectedSamples samplesPerClass row
    
    % Divide training set into train and validation subsets
    rand('state',0);
    k=randperm(length(tempdata));
    traindata=tempdata(k(1:floor(length(tempdata)*5/6)),:);
    valdata=tempdata(k(floor(length(tempdata)*5/6)+1:end),:);
    trainlabels=templabels(k(1:floor(length(tempdata)*5/6)));
    vallabels=templabels(k(floor(length(tempdata)*5/6)+1:end));
    fprintf('Train \t Val\n');
    for labeliter=1:5;
        fprintf('%i \t %i\n', sum(trainlabels==labeliter), sum(vallabels==labeliter));
    end
    clear tempdata templabels k
    
    % ---------------------------- Train --------------------------------------
    layerSize = [200 200]; % 200-200 used in article
    
    % Parameters
    % Initial hidden biases can be changed in NNLayer.m row 498
    rbmParams.numEpochs = 300; %300 used in article
    rbmParams.verbosity = 1;
    rbmParams.miniBatchSize = 1000;
    rbmParams.attemptLoad = 0;
    % set DBN params
    dbnParams.numEpochs = 50; % 50 used in article
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
    [topActivs, a] = dnnObj.PropLayerActivs(data);
    [~,ytrain] = max(topActivs,[],2);
    ytrain=single(ytrain);
    
    % Train HMM on inference from train data
    [A, B] = hmmestimate(ytrain, labels, 'PSEUDOTRANSITIONS', 0.001*ones(5, 5), 'PSEUDOEMISSIONS',0.001*ones(5, 5));
    
    %Inference on test data
    [topActivs2, ~] = dnnObj.PropLayerActivs(testdata);
    [~, ytest] = max(topActivs2,[],2);
    ytest=single(ytest);
    
    % Test results from HMM
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
    
    % Save after each iteration
    save resultsRAW results
end
