
% train a DBN on a subset of the MNIST handwritten digits

% make sure we can see the code
addpath ../lib

% load the dataset, a subset of the MNIST digits
load mnist_small.mat

% train a reconstruciton DBN with layer sizes 1000-500-250-30 (same as in Hinton
% and Salakhutdinov 2006 Science paper)

% set RBM params (could make into a cell array for each RBM if we wanted
% different params)
rbmParams.strID = 'recon'; % used in model filename

rbmParams.outDir = ['out' filesep];	% where the file will be written

rbmParams.numEpochs = 10; % number of complete passes through data, usually want more 
												  % than this

rbmParams.verbosity = 1; % means that model is written to file but epoch error
			 									 % written to console

% set DBN params
dbnParams.strID = 'recon';
dbnParams.outDir = ['out' filesep];	
dbnParams.numEpochs = 10;
dbnParams.verbosity = 1;


fprintf('Training reconstruction DBN...this may take a while.\n\n');

% note: DBN training takes a while; this example may take ~10-20 mins to finish
reconDBN = TrainDeepNN([1000 500 250 30],'RBM',rbmParams,dbnParams, ...
    trainSamples,valSamples);


% show a few originals and reconstructions
recon = reconDBN.PropLayerActivs(testSamples(1:5,:));
figure
for i = 1:5
	subplot(5,2,(i-1)*2+1);
	title('original');
	imshow(reshape(testSamples(i,:),28,28)');
	subplot(5,2,(i-1)*2+2);
	title('recon');
	imshow(reshape(recon(i,:),28,28)');
end
