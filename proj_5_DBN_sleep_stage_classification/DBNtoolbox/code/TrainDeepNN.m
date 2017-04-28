function dnnObj = TrainDeepNN(layerSizes, layerType, nnlParams, dnnParams, ...
	trainSamples, valSamples, varargin)
%
% TrainDeepNN trains a deep neural network first training individual neural net
% layers (GreedyLayerTrain.m) and then stacks them into a deep neural net and 
% trains them via backpropogation.
%
%
% USAGE 
%   
%   dnnObj = TrainDeepNN(layerSizes, layerType, nnlParams, dnnParams, ...
%							trainSamples, valSamples, [trainLabels, valLabels])
%
%
% INPUTS 
%
%   layerSizes :	a vector indicating the number of hidden units in each layer
%
%   layerType :		a string denoting the child class of NNLayer used for each
%									layer
%
%   nnlParams :		the params struct passed to the child class of NNLayer
%
%		dnnParams :		the params struct passed to the DeepNN class
%
%		trainSamples :	an N-by-D matrix of the N training samples with D dimensions
%
%		valSamples :	an M-by-D matrix of the M validation samples with D dimensions
%
%		trainLabels :	(optional) an N-by-C matrix of the N training labels for C 
%									classes (correct class is 1, all others 0)
%
%		valLabels :		(optional) an N-by-C matrix of the N training labels for C 
%									classes (correct class is 1, all others 0)
%
%
% OUTPUTS 
%
%   dnnObj :			the DeepNN object produced after training
%
%
% EXTERNAL FILE REQUIREMENTS
%
%   NNLayer.m,
%		the class identified by 'layerType' (e.g. RBM.m),
%		GreedyLayerTrain.m,
%		DeepNN.m
%
%
% KNOWN INCOMPATIBILITIES 
%
%		None
%
%
% ACKNOWLEDGEMENTS 
%
%   None
%
%
% CONTRIBUTORS 
%
%	Created by:
%   	Drausin Wulsin
%   	09/08/2010
%  	Translational Neuroengineering Lab, University of Pennsylvania
%
%
%


% CONSTANTS 
% -------------------------------------------------------------------------

% -------------------------------------------------------------------------


% INPUT/FILE PROCESSING
% -------------------------------------------------------------------------
if length(varargin) >= 2
	trainLabels = varargin{1};
	valLabels = varargin{2};
end
% -------------------------------------------------------------------------


% MAIN
% -------------------------------------------------------------------------

% do greedy layer-wise pretraining
nnLayers = GreedyLayerTrain(trainSamples, valSamples, layerSizes, layerType, ...
	nnlParams);


% construct DeepNN and train with backprop
dnnObj = DeepNN(nnLayers, dnnParams);

fprintf('Training DeepNN...\n');

if length(varargin) >= 2 % classification NN
	dnnObj.Train(trainSamples, valSamples, trainLabels, valLabels);
else
	dnnObj.Train(trainSamples, valSamples); % reconstruction NN
end

fprintf('Finished training DeepNN.\n\n');

% -------------------------------------------------------------------------

end

% SUBFUNCTIONS
% -------------------------------------------------------------------------

% -------------------------------------------------------------------------

