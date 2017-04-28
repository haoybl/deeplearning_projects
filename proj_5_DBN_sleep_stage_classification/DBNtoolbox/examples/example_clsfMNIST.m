
% train a DBN on a subset of the MNIST handwritten digits

% make sure we can see the code
addpath ../lib

% load the dataset, a subset of the MNIST digits
load mnist_small.mat

% train a reconstruciton DBN with layer sizes 1000-500-250-30 (same as in Hinton
% and Salakhutdinov 2006 Science paper)

% set RBM params (could make into a cell array for each RBM if we wanted
% different params)
rbmParams.strID = 'clsf'; % used in model filename

rbmParams.outDir = ['out' filesep];	% where the file will be written

rbmParams.numEpochs = 10; % number of complete passes through data, usually want more 
												  % than this

rbmParams.verbosity = 1; % means that model is written to file but epoch error
			 									 % written to console

% set DBN params
dbnParams.strID = 'clsf';
dbnParams.outDir = ['out' filesep];	
dbnParams.numEpochs = 10;
dbnParams.verbosity = 1;


fprintf('Training classificatoin DBN...this may take a while.\n\n');

% note: DBN training takes a while; this example may take ~10-20 mins to finish
clsfDBN = TrainDeepNN([500 500 2000],'RBM',rbmParams,dbnParams, ...
    trainSamples,valSamples,trainLabels,valLabels);



% show some output

% make label predictions, subtract 1 b/c labels start at 1 but index starts at 1
[~,predLabels] = max(clsfDBN.PropLayerActivs(testSamples(1:5,:)),[],2); 
predLabels = predLabels - 1;

figure;
for i = 1:5
	subplot(5,1,i);
	imshow(reshape(testSamples(i,:),28,28)');
	title(sprintf('pred label: %d',predLabels(i)));
end


