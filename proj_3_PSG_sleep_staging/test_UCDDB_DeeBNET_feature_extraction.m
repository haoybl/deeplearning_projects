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
% RBM3
rbmParams=RbmParameters(500,ValueType.binary);
rbmParams.maxEpoch=maxEpoch;
rbmParams.samplingMethodType=SamplingClasses.SamplingMethodType.CD;
rbmParams.performanceMethod='reconstruction';
dbn.addRBM(rbmParams);
% RBM4
rbmParams=RbmParameters(250,ValueType.binary);
rbmParams.maxEpoch=maxEpoch;
rbmParams.samplingMethodType=SamplingClasses.SamplingMethodType.CD;
rbmParams.performanceMethod='reconstruction';
dbn.addRBM(rbmParams);
% RBM5
rbmParams=RbmParameters(3,ValueType.gaussian);
rbmParams.maxEpoch=maxEpoch;
rbmParams.samplingMethodType=SamplingClasses.SamplingMethodType.CD;
rbmParams.performanceMethod='reconstruction';
dbn.addRBM(rbmParams);

dbn.train(data);
dbn.backpropagation(data);


