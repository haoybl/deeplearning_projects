function nnLayers = GreedyLayerTrain(trainSamples, valSamples, layerSizes, ...
	layerType, varargin) %#ok<STOUT>
%
% This function greedily trains successive neural net layers of the type
% indiciated in 'layerType'
%
%
% USAGE 
%   
%   nnLayers = FunctionName(trainingSamples, layerSizes, layerType, [params])
%
%
% INPUTS 
%
%   trainSamples :	an N-by-D matrix representing N training samples with
%										D dimensions
%
%   layerType :			a string denoting child class of NNLayer used for each
%										layer
%
%		layerSizes :		the number of hidden units for each layer
%
%   params :				(optional) a struct containing the params to pass to each 
%										layer class
%
%
% OUTPUTS 
%
%   nnLayers :			an array of trained layerType objects for each layer
%
%
%
% EXTERNAL FILE REQUIREMENTS
%
%   NNLayer.m, 
%		the class identified by 'layerType' (e.g. RBM.m)
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
%   	02/08/2010
%  	Translational Neuroengineering Lab, University of Pennsylvania
%
%


% CONSTANTS 
% -------------------------------------------------------------------------

% -------------------------------------------------------------------------


% INPUT/FILE PROCESSING
% -------------------------------------------------------------------------

% init params 
if nargin >= 4
	params = varargin{1};
else
	params = struct;
end

% determine if the inputs are continueous or binary
isInputCont = min(min(trainSamples)) < 0 || max(max(trainSamples)) > 1;

% make layer params for each layer if params isn't a cell array (i.e. same
% params for all)
if iscell(params)
	layerParams = params; 
else
	for i = 1:length(layerSizes)
		layerParams{i} = params; %#ok<AGROW>
	end
end

% set string IDs base for each layer
if ~isfield(layerParams{1},'strID')
	strIDbase = 'nnl';
else
	strIDbase = layerParams{1}.strID;
end

% -------------------------------------------------------------------------


% MAIN
% -------------------------------------------------------------------------

curTrainInput = trainSamples;
curValInput = valSamples;

% loop through each layer
for i = 1:length(layerSizes)

	fprintf('Training layer %d...\n',i);

	% set this layer's strID
	layerParams{i}.strID = sprintf('%s.%d',strIDbase,i);
	
	% init this layer
	if i == 1		
		if strcmp(layerType,'FRBM')
			eval(sprintf('nnLayers(i) = %s(%d,%d,curTrainInput,layerParams{i});', ...
				layerType, size(trainSamples,2),layerSizes(i)));
		else
			eval(sprintf('nnLayers(i) = %s(%d,%d,layerParams{i});', layerType, ...
				size(trainSamples,2),layerSizes(i)));
		end
	else
		if strcmp(layerType,'FRBM')
			eval(sprintf('nnLayers(i) = %s(%d,%d,curTrainInput,layerParams{i});', ...
				layerType, layerSizes(i-1),layerSizes(i)));
		else
			eval(sprintf('nnLayers(i) = %s(%d,%d,layerParams{i});', layerType, ...
				layerSizes(i-1),layerSizes(i)));
		end
	end
	
	% train
	if i == 1 && isInputCont
		% make visisble layer handle continuous-valued inputs (usually requires
		% much slower learning rate)
		nnLayers(i).Train(curTrainInput,curValInput,0,isInputCont); 
	else
		nnLayers(i).Train(curTrainInput,curValInput);
	end

	fprintf('Finished layer %d training\n\n',i);
	
	% update the current input for next layer
	if i < length(layerSizes)
		curTrainInput = nnLayers(i).PropHidFromVis(curTrainInput);
		
		if ~isempty(curValInput)
			curValInput = nnLayers(i).PropHidFromVis(curValInput);
		end
	end
end

% -------------------------------------------------------------------------

end

% SUBFUNCTIONS
% -------------------------------------------------------------------------

% -------------------------------------------------------------------------

