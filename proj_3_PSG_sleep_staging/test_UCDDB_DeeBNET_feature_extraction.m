%% TEST_UCDDB_DEEBNET_FEATURE_EXTRACTION
% This script is mainly used to test DeeBNet library for extracting useful
% features from UCDDB PSG database.

clear;
clc;

%% Import DeeBNet Library
addpath(genpath('resources'));

%% Load and prepared data in proper format
load('UCDDB_data_prepared_for_training_no_onehot.mat');

data=DataClasses.DataStore();
% Data value type is gaussian because the value can be consider a real
% value [-Inf +Inf]
data.valueType=ValueType.gaussian;

data.trainData=trainData;
data.trainLabels=trainLabel;
data.testData=testData;
data.testLabels=testLabel;
data.normalize('meanvar');
data.validationData=validData;
data.validationLabels=validLabel;
data.shuffle();

clear('trainData', 'trainLabel', 'testData', 'testLabel', 'validData', 'validLabel');
fprintf(1,'Training Data Prepared, Initializing DBN...\n');

%% Initialize DBN

dbn=DBN();
dbn.dbnType='autoEncoder';
maxEpoch=200;

% RBM1
rbmParams=RbmParameters(200,ValueType.binary);
rbmParams.maxEpoch=maxEpoch;
rbmParams.samplingMethodType=SamplingClasses.SamplingMethodType.FEPCD;
rbmParams.performanceMethod='reconstruction';
dbn.addRBM(rbmParams);
% RBM2
rbmParams=RbmParameters(200,ValueType.binary);
rbmParams.maxEpoch=maxEpoch;
rbmParams.samplingMethodType=SamplingClasses.SamplingMethodType.CD;
rbmParams.performanceMethod='reconstruction';
dbn.addRBM(rbmParams);

% RBM3 For plot
rbmParams=RbmParameters(3,ValueType.gaussian);
rbmParams.maxEpoch=maxEpoch;
rbmParams.samplingMethodType=SamplingClasses.SamplingMethodType.CD;
rbmParams.performanceMethod='reconstruction';
dbn.addRBM(rbmParams);

dbn.train(data);
dbn.backpropagation(data);

save('training_result.mat', 'dbn', 'data');


%% plot
figure;
%plotFig=[{'mo' 'go' 'm+' 'r+' 'ro' 'k+' 'g+' 'ko' 'bo' 'b+'}];
plotFig=[{'mo' 'go' 'm+' 'r+' 'ro'}];
for i=0:4
    img=data.testData(data.testLabels==i,:);
    ext=dbn.getFeature(img);
    plot3(ext(:,1),ext(:,2),ext(:,3),plotFig{i+1});hold on;
end
legend('Wake','REM','S1','S2','SWS');
hold off;
