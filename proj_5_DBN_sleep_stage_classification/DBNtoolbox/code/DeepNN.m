classdef DeepNN < handle 
%
% DeepNN is a class representing a deep neural network, intialized through
% either individual neural net layer training (i.e. RBM) or random 
%	initialization and trained with stochastic conjugate gradient descent 
%
% CONSTRUCTORS
%   
%		obj = DeepNN(layerSizes, [params])
%		- random initialization based on the 'layerSizes' matrix, where each row 
%		represents a layer (including the input layer) and contains the size of the 
%		layer in the first column and a 0/1 value in the second column identifying 
%		whether the layer is binary or linear; 'params' is the optional parameters 
%		struct
%
%		obj = DeepNN(NNLayers, [params])
%		- pretraining intialization based on the objects in 'NNLayer' array of 
%		pretrained NNLayer (and its subclasses, like RBM) objects; 'params' is the 
%		optional parameters struct
%		
%		obj = DeepNN(weights, biases, hidLinear, [params])
%		- explicit initialization via the cell arrays 'weights' and 'biases' and
%		vector 'hidLinear'; 'params' is the optional parameters struct
%		
%
%
% PUBLIC PROPERTIES [Public Read-Only, Protected Read/Write]
%
%		weights :					a cell array of the weights matrices between each layer
%
%		biases :					a cell array of the bias vectors for each layer
%
%		hidLinear :				a vector of logical (0/1) values indicating whether that
%											layer is linear real-valued (vs. nonlinear [0,1])
%		
%		visLinear :				a single logical value (0/1) value indicating whether the
%											input data is linear real-valued (vs. nonlinear [0,1]);
%											slightly redundant since same value as hidLinear(end)
%		
%		isClassifier :		a logical value (0/1) that describes whether the DBN is
%											trained to produce label outputs
%		
%		weightsHistory :	a cell array containing the weights at each epoch, where 
%											weights are intermittently stored according to the 
%											obj.histLogFreq and obj.overfitHistory properties
%		
%		biasesHistory :		same as the weightsHistory, except for the biases
%		
%		errorHistory :		a struct array describing the errors calulated at each
%											epoch with fields '.time','.trainRMSE','.trainCEE',
%											'.valRMSE', and '.valCEE'
%		
%		statusFile :			the file path of the status file, which mostly gets epoch
%											errors values printed to it (default:
%											outDir/dnn_status.txt')
%
%		modelFile :				the file path of the model file, which contains the saved
%											DeepNN class object as the variable 'dnnObj'; saved after
%											each training epoch (default: outDir/dnn_obj.mat)
%
%
% PUBLIC METHODS
%
%		Train(trainData, valData, [trainLabels], [valLabels])
%		- train the DeepNN using the training and validation data; trainData and
%		valData are either 2D matrices (rows: samples, cols: dimensions) or 1D cell
%		arrays of filepaths containing the data; trainLabels and valLabels are
%		optional 1D vectors that make the training supervised
%
%		[topActivs, layerActivs] = PropLayerActivs(layerInput, [startLayer], [endLayer])
%		- propogate the 'layerInpout' (rows: samples, cols: dims) through the network,
%		optionally starting at 'startLayer' and ending at 'endLayer'; 'topActivs' is
%		a 2D matrix of the top (end) layer activiations (either reconstruction or
%		labels output), and 'layerActivs' is a cell array of matrices for each
%		layer's activations (including the top layer)
%		
%		SetParams(params)
%		- sets the parameters of the model (as done in the constructor)
%
%		SetInternalParams()
%		- sets the filename (statusFile and modelFile) based on the obj.outDir
%		parameter
%
%		SetModel(obj, weights, biases, hidLinear)
%		- manually set the weights, biases, and hidLinear vars of the model
%
%		[STATIC] y = Sigm(x)
%		- the logistic sigmoid function, y(x) = 1 / (1 + exp(-x))
%
%		[STATIC] y = Softmax(x)
%		- the softmax function over the vector x, y(x) = exp(x)/sum(exp(x))
%
%		[STATIC] [weightsVect,weightsDims] = Weights2Vect(weights, biases)
%		- takes the weights and biases of the layers of the model and converts them
%		to a 1D vector 'weightsVect' and Nx2 matrix 'weightsDims', which describes 
%		the architecture of the N layers
%
%		[STATIC] [weights, biases] = Vect2Weights(weightsVect,weightsDims)
%		- takes 1D 'weightsVect' and Nx2 matrix 'weightsDims' arguments and converts
%		them back into the cell arrays of weights and biases
%
%		[STATIC] lblMat = GetLabelsMat(labelsVect)
%		- converts the Mx1 dimensional 'labelsVect' of C (integer) classes into an
%		MxC zeros matrix of labels with a single one in in each row for the correct
%		class
%
%
% EXTERNAL CLASS AND FILE REQUIREMENTS
%
%   minimize.m (by Carl Edward Rasmussen) 
%		GetError.m
%
%
% KNOWN INCOMPATIBILITIES 
%
%		None
%
%
% ACKNOWLEDGEMENTS 
%
%   This code is heavily adapted from the backprop.m and backpropclassify.m
%   scripts written by Ruslan Salakhutdinov and Geoff Hinton, whose complete
%		code can be found at http://www.cs.toronto.edu/~hinton/
%		MatlabForSciencePaper.html
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
%
%
% TODO
% - add Contrastive Wake-Sleep training
      
    
	% PUBLIC PROPERTIES ---------------------------------------------------
	properties (GetAccess = public, SetAccess = protected)
		
		% a cell array of the weights matrices between each layer
		weights

		% a cell array of the bias vectors for each layer
		biases

		% a vector of logical (0/1) values indicating whether that layer is linear 
		% real-valued (vs. nonlinear [0,1])
		hidLinear
		
		% a single logical value (0/1) value indicating whether the input data is 
		% linear real-valued (vs. nonlinear [0,1]); slightly redundant since same 
		% value as hidLinear(end)
		visLinear
		
		% a logical value (0/1) that describes whether the DBN is	trained to produce
		% label outputs
		isClassifier = 0;
		
		% a cell array containing the weights at each epoch, where weights are 
		% intermittently stored according to the obj.histLogFreq and 
		% obj.overfitHistory properties
		weightsHistory
		
		% same as the weightsHistory, except for the biases
		biasesHistory
		
		% a struct array describing the errors calulated at each epoch with fields 
		% time','.trainRMSE','.trainCEE','.valRMSE', and '.valCEE'
		errorHistory

		% the file path of the status file, which mostly gets epoch errors values 
		% printed to it (default:	outDir/dnn_status.txt')
		statusFile

		% the file path of the model file, which contains the saved	DeepNN class 
		% object as the variable 'dnnObj'; saved after each training epoch 
		% (default: outDir/dnn_obj.mat)
		modelFile
		
	end
	    
	% PROTECTED PROPERTIES ------------------------------------------------
	properties (Access = protected)
		
		% TUNABLE PARAMETERS: set any of these parameters by setting a field with
		% their same name in the 'params' struct, which is passed either into the
		% constructor or to the obj.SetParams(params) function

		% the string identifier of the model, used in model and status filenames
		strID = '';
		
		% the number of training epochs
		numEpochs = 50;

		% (if DeepNN is a classifier) the number of epochs to just train the top 
		% logistic regression layer before training	the rest of the network 
		numTopInitEpochs = 5;

		% the number of samples to combine into one	mini-batch
		miniBatchSize = 1000;

		% the number of conjugate-gradient linesearches to do
		cgMaxSearches = 3;
		
		% a logical scalar of whether to check for an overfit model	and stop 
		% training if one is found
		checkOverfit = 1;
		
		% the number of validation epochs to use in checking for overfit
		overfitHistory = 10;
		
		% the how often weights/biases are stored after each epoch
		historyLogFreq = 10;
		
		% (logical) whether the DBN to to be trained online (single pass through
		% data)
		trainOnline = 0;
		
		% used in online training (where data is loaded from files rather than 
		% passed in to the 'Train' function), the number of samples in each segment 
		% file (MUST be set in the params struct passed into the DeepNN constructor)
		online_numSampsPerSeg
				
		% the verbosity level during training:
		%	0 - no epoch output
		%	1 - epoch error written to console and model saved to file
		%	2 - epoch error and model written to files
		verbosity = 2;
		
		% the directory to write file output
		outDir = '';
		
		% whether to attempt to load an existing obj with the same model filename if
		% one exists
		attemptLoad = 1;

		% other properties
		% flag if there is a fatal error (0 = ok, 1 = fatal error)
		fatalFlag = 0;
				
	end
    
	% PRIVATE PROPERTIES --------------------------------------------------
	properties (Access = private)
	end
    
    
  % PUBLIC METHODS ------------------------------------------------------
	methods (Access = public)

		% CONSTRUCTOR
		%
		%	obj = DeepNN(layerSizes, [params])
		%	- random initialization based on the 'layerSizes' matrix, where each row 
		%	represents a layer (including the input layer) and contains the size of  
		%	the layer in the first column and a 0/1 value in the second column  
		%	identifying whether the layer is binary or linear; 'params' is the 
		%	optional parameters struct
		%
		%	obj = DeepNN(NNLayers, [params])
		%	- pretraining intialization based on the objects in 'NNLayer' array of 
		%	pretrained NNLayer (and its subclasses, like RBM) objects; 'params' is the 
		%	optional parameters struct
		%		
		%	obj = DeepNN(weights, biases, hidLinear, [params])
		%	- explicit initialization via the cell arrays 'weights' and 'biases' and
		%	vector 'hidLinear'; 'params' is the optional parameters struct
		%
		function obj = DeepNN(varargin)

			% VALIDATE ARGUMENTS ------------------------------------------

			% need to have at least one argument
			if nargin == 0
				m = MException('Constructor:InvalidArgs',['You must at least ' ...
					'have one argument defining the NNLayer objects or layer ' ...
					'sizes.']);
				m.throw();
			end

			% arguments must either be an array of NNLayer objects, an
			% array of layer sizes, or the weights, biases, and hidLinear params
			if nargin >= 1
				validateattributes(varargin{1},{'RBM','AutoEnc','cell','numeric'}, ...
					{'nonempty'});
			end

			if nargin >= 2
				validateattributes(varargin{2},{'cell','struct','numeric'},{'nonempty'});
			end

			if nargin >= 3
				validateattributes(varargin{3},{'numeric'},{'nonempty'});
			end

			if nargin >= 4
				validateattributes(varargin{4},{'struct'},{'nonempty'});
			end

			% END VALIDATE ARGUMENTS --------------------------------------
			
			
			% if passing in model params, set them
			if nargin >= 3
				obj.weights = varargin{1};
				obj.biases = varargin{2};
				obj.hidLinear = varargin{3};
				% assume it's not a classifier if the number if output units equals number of
				% input units
				obj.isClassifier = size(obj.weights{1},1) ~= size(obj.weights{end},2);
			else
				obj.Init(varargin{1});
			end
			
			% set internal params
			obj.SetInternalParams();

			% set any parameters passed in
			if nargin == 2 
				obj.SetParams(varargin{2});
			elseif nargin == 4
				obj.SetParams(varargin{4});
			else
				obj.SetParams(struct); % pass empty struct
			end

			% set internal params
			obj.SetInternalParams();
			
			% attempt to load the model
			obj = obj.AttemptLoadModel();

		end

		%	Train(trainData, valData, [trainLabels], [valLabels])
		%	- train the DeepNN using the training and validation data; trainData and
		%	valData are either 2D matrices (rows: samples, cols: dimensions) or 1D cell
		%	arrays of filepaths containing the data; trainLabels and valLabels are
		%	optional 1D vectors that make the training supervised
		function Train(obj, trainData, valData, varargin)

			% if varargin is nonempty, then its value is the trainLabels
			% vector, which we turn into matrices
			if length(varargin) >= 2
				trainLabels = DeepNN.GetLabelsMat(varargin{1});
				valLabels = DeepNN.GetLabelsMat(varargin{2});
				if ~obj.isClassifier
					obj.MakeClassifier(size(trainLabels,2));
				end
			end
			
			% make dummy val data
			if ~exist('valData','var')
				valData = [];
			end

			% start log and timer
			if ~obj.isClassifier
				obj.MsgOut(['Epoch\tDate-Timestamp\t\tAvgTrainCEE\tAvgTrainRMSE' ...
					'\t\tAvgValCEE\tAvgValRMSE\n0\t' datestr(now) '\t\t0\t0\t\t0\t0\n'])
			else
				obj.MsgOut(['Epoch\tDate-Timestamp\t\tAvgTrainCEE\tAvgTrainRMSE' ...
					'\tAvgTrainPerc\t\tAvgValCEE\tAvgValRMSE\tAvgValPerc\n0\t' ...
					datestr(now) '\t\t0\t0\t0\t\t0\t0\t0\n'])
			end
			tic;

			isOverfit = 0;

			% epochs loop, start after last finished epoch
			for e = (length(obj.errorHistory)+1):obj.numEpochs
				
				% make sure we're not overfit
				if isOverfit
					break
				end
				
				% get the current train/valSamples (no labels as of yet) for either offline
				% or online learning
				[trainSamples,valSamples] = obj.GetSamples(trainData,valData,e);
				
				% check for fatal flag
				if obj.fatalFlag
					obj.MsgOut('Fatal Error: training aborted\n',2);
					return
				end
				
				% calc number of mini-batches to use
				numTrainMBs = floor(size(trainSamples,1) / obj.miniBatchSize);
					
				for mb = 1:numTrainMBs

					% get the current mini-batchs
					curTrainMB = trainSamples((mb-1)*obj.miniBatchSize + ...
						(1:obj.miniBatchSize),:);
					
					% update the network weights/biases
					if obj.isClassifier
						
						curTrainLabelsMB = trainLabels((mb-1)*obj.miniBatchSize + ...
							(1:obj.miniBatchSize),:);
						
						obj.Update(e,curTrainMB,curTrainLabelsMB);
						
					else
						
						obj.Update(e,curTrainMB);
						
					end
					
				end
				
				% log epoch
				if ~obj.isClassifier
					obj.LogEpoch(e,trainSamples,valSamples)
				else
					obj.LogEpoch(e,trainSamples,valSamples,trainLabels,valLabels)
				end

				% check to see if model is overfit
				if obj.checkOverfit && e > obj.overfitHistory
					isOverfit = obj.CheckOverfit();
				end

				% save this object
				dnnObj = obj; %#ok<NASGU>

				save(obj.modelFile,'dnnObj');

			end

		end
		
		%	[topActivs, layerActivs] = PropLayerActivs(layerInput, [startLayer], ...
		%		[endLayer])
		%	- propogate the 'layerInpout' (rows: samples, cols: dims) through the 
		%	network, optionally starting at 'startLayer' and ending at 'endLayer'; 
		%	'topActivs' is a 2D matrix of the top (end) layer activiations (either 
		%	reconstruction or labels output), and 'layerActivs' is a cell array of 
		%	matrices for each layer's activations (including the top layer)
		function [topActivs, layerActivs] = PropLayerActivs(obj, layerInput, ...
				varargin)

			% calc the starting layer
			if length(varargin) >= 1
				startLayer = varargin{1};
			else
				startLayer = 1;
			end

			% calc the ending layer
			if length(varargin) >= 2
				endLayer = varargin{2};
			else
				endLayer = length(obj.weights);
			end

			% propogate the test sample activiations through the network
			layerActivs = cell(1,endLayer-startLayer);
			for i = startLayer:endLayer
				c = i - startLayer + 1;
				if obj.hidLinear(i)
					layerActivs{c} = layerInput * obj.weights{i} + ...
						repmat(obj.biases{i},size(layerInput,1),1);
				else
					layerActivs{c} = DeepNN.Sigm(layerInput * obj.weights{i} + ...
						repmat(obj.biases{i},size(layerInput,1),1));
				end
				layerInput = layerActivs{c};
			end

			% if the DeepNN is a classifier, softmax the last layer's outputs
			if obj.isClassifier && endLayer == length(obj.weights)
				layerActivs{end} = obj.Softmax(layerActivs{end});
			end

			% for convenience
			topActivs = layerActivs{end};

		end
		
		%	SetParams(params)
		%	- sets the parameters of the model (as done in the constructor)
		function SetParams(obj,params) 
			
			% get the parameters set in 'params'
			fields = fieldnames(params);
			
			% for each parameter field
			for fi = 1:length(fields)
				
				% build command to set param
				cmd = sprintf('obj.%s = params.%s;',fields{fi},fields{fi});
				
				% execute command
				try
					eval(cmd);
				catch e
					obj.MsgOut(sprintf(['Error setting param %s : %s Using ' ...
						'default value.\n'], fields{fi}, e.message),1);
				end
			end
			
			% make sure outDir has proper filesep at end
			if ~isempty(obj.outDir) && ~strcmp(obj.outDir(end),filesep)
				obj.outDir = [obj.outDir filesep];
			end
			
			% make sure outDir exists
			if ~isempty(obj.outDir) && ~exist(obj.outDir,'dir')
				mkdir(obj.outDir);
			end
			
		end
		
		%	SetInternalParams()
		%	- sets the filename (statusFile and modelFile) based on the obj.outDir
		%	parameter
		function SetInternalParams(obj)
			
			if isempty(obj.strID)
				obj.strID = 'dnn';
			else
				obj.strID = [obj.strID '.dnn'];
			end
			
			% set the rbm file path
			obj.modelFile = [obj.outDir obj.strID '_obj.mat'];
			
			% set the output status file path, if necessary
			if obj.verbosity == 2
				obj.statusFile = [obj.outDir obj.strID '_status.txt']; 
			end
			
		end
		
		% SetModel allows user to explicitly set internal params of the model
		function SetModel(obj, weights, biases, hidLinear)
			obj.weights = weights;
			obj.biases = biases;
			obj.hidLienar = hidLinear;
			obj.isClassifier = size(obj.weights{1},1) ~= size(obj.weights{end},2);
		end
        
	end
	
	methods (Access = public, Static)
		
		%	[STATIC] y = Sigm(x)
		%	- the logistic sigmoid function, y(x) = 1 / (1 + exp(-x))
		function y = Sigm(x)
			y = 1 ./ (1 + exp(-x));
		end
		
		%	[STATIC] y = Softmax(x)
		%	- the softmax function over the vector x, y(x) = exp(x)/sum(exp(x))
		function y = Softmax(x)
			e = exp(x);
			y = e ./ repmat(sum(e,2),1,size(x,2));
		end
		
		%	[STATIC] [weightsVect, weightsDims] = Weights2Vect(weights, biases)
		%	- takes the weights and biases of the layers of the model and converts 
		%	them to a 1D vector 'weightsVect' and Nx2 matrix 'weightsDims', which  
		%	describes the architecture of the N layers
		function [weightsVect,weightsDims] = Weights2Vect(weights, biases)
				
			weightsVect = [];
			%weightsVect = zeros(1,vectLen); 
			weightsDims = zeros(length(weights),2);
			
			% loop through each layer
			for i = 1:length(weights)
				weightsVect = [weightsVect weights{i}(:)' biases{i}]; %#ok<AGROW>
				weightsDims(i,:) = [size(weights{i},1) size(weights{i},2)];
			end
			
		end
		
		%	[STATIC] [weights, biases] = Vect2Weights(weightsVect,weightsDims)
		%	- takes 1D 'weightsVect' and Nx2 matrix 'weightsDims' arguments and 
		%	converts them back into the cell arrays of weights and biases
		function [weights, biases] = Vect2Weights(weightsVect,weightsDims)
			
			weights = cell(1,size(weightsDims,1));
			biases = cell(1,size(weightsDims,1));
			cursor = 1;
			
			% for each layer
			for i = 1:size(weightsDims,1)
				layerLen = weightsDims(i,1)*weightsDims(i,2) + weightsDims(i,2);
				layerVect = weightsVect(cursor : cursor + layerLen - 1);
				cursor = cursor + layerLen;
				
				% reconstruct weights and biases
				weights{i} = reshape(layerVect(1:weightsDims(i,1) ...
					* weightsDims(i,2)), weightsDims(i,1),weightsDims(i,2));
				biases{i} = layerVect(weightsDims(i,1)*weightsDims(i,2)+ ...
					1:end)';
			end
		end	
		
		%	[STATIC] lblMat = GetLabelsMat(labelsVect)
		%	- converts the Mx1 dimensional 'labelsVect' of C (integer) classes into an
		%	MxC zeros matrix of labels with a single one in in each row for the 
		%	correct class
		function lblMat = GetLabelsMat(labelsVect)
			
			lblMat = zeros(length(labelsVect),max(labelsVect));
			for i = 1:length(labelsVect)
				lblMat(i,labelsVect(i)) = 1;
			end
			
		end
		
	end
	
	% PROTECTED METHODS ---------------------------------------------------
	methods (Access = protected)
		
		%	MsgOut(text, [isError])
		%	- outputs string in 'text' to either console or a file,	depending on 
		%	verbosity properties; 'isError' is optional but if its value is > 0, 'text' 
		%	will be printed to console even with verbosity 0; if 'isError' value is > 
		%	1, 'obj.fatalFlag' is set to 1	
		function MsgOut(obj, text, isError)
			
			% set isError to 0 if it doesn't exist
			if ~exist('isError','var')
				isError = 0;
			end
			
			if obj.verbosity == 1 || (isError && obj.verbosity == 0)
				fprintf(text)
			elseif obj.verbosity == 2 
				fid = fopen(obj.statusFile,'a');
				fprintf(fid,text);
				fclose(fid);
			end
			
			% set fatalFlag if necessary
			if isError > 1
				obj.fatalFlag = 1;
			end
		end
		
		%	Init(varargin)
		%	- initializes all the model properties with either a cell array of
		%	NNLayer objects (for pretraining initialization) or an Mx2 matrix
		%	for M layers, including the input layer, where the first column is the
		%	layer size and the second column is where the layer is linear or not
		function Init(obj, varargin)
			
			% if we've already init'd the weights, biases, and hidLinear, do nothing
			if ~isempty(obj.weights)
			
			% initialize from previously-trained NNLayer objects
			elseif ~isnumeric(varargin{1})
				
				% init the weights and biases cell arrays
				obj.weights = cell(1,2*length(varargin{1}));
				obj.biases = cell(1,2*length(varargin{1}));

				% init hidLinear flags for whether a hidden layer is linear 
				obj.hidLinear = zeros(1,2*length(varargin{1}));
				
				layerObjs = varargin{1};
				
				% use each NNLayer object to init weights and biases
				for i = 1:length(layerObjs)
					[w, hb, vb] = layerObjs(i).GetModel();
					
					% use NNLayer weights and their transpose 
					
					% encoding side
					obj.weights{i} = w;
					obj.biases{i} = hb;
					
					% decoding side
					obj.weights{end-i+1} = w';
					obj.biases{end-i+1} = vb;
					
					% determine if layers are linear
					if i == 1 && layerObjs(i).visLinear
						obj.visLinear = layerObjs(i).visLinear;
						obj.hidLinear(end) = layerObjs(i).visLinear;
					end
					obj.hidLinear(i) = layerObjs(i).hidLinear;
					obj.hidLinear(end-i) = layerObjs(i).hidLinear;
					
				end
				
			% initialize randomly
			else
				
				% init the weights and biases cell arrays
				obj.weights = cell(1,2*(size(varargin{1},1)-1));
				obj.biases = cell(1,2*(size(varargin{1},1)-1));

				% init hidLinear flags for whether a hidden layer is linear 
				obj.hidLinear = zeros(1,2*(size(varargin{1},1)-1));
				
				layerSizes = varargin{1}(:,1);
				isLinear = varargin{1}(:,2);
				
				% determine if visible input is linear
				if isLinear(1)
					obj.visLinear = isLinear(1);
					obj.hidLinear(end) = isLinear(1);
				end
				
				% use the layer sizes to randomly init the weights and biases
				for i = 2:length(layerSizes)
					
					% define init spread for this layer					
					initSpread = 4*sqrt(6/(layerSizes(i-1) + layerSizes(i)));

					% encoding weights and biases
					obj.weights{i-1} = -initSpread + 2*initSpread * ...
						rand(layerSizes(i-1),layerSizes(i));
					obj.biases{i-1} = zeros(1,layerSizes(i));

					% decoding weights and biases
					obj.weights{end-i+2} = -initSpread + 2*initSpread * ...
						rand(layerSizes(i),layerSizes(i-1));
					obj.biases{end-i+2} = zeros(1,layerSizes(i-1));
					
					% determine if hidden layers are linear
					obj.hidLinear(i-1) = isLinear(i);
					obj.hidLinear(end-i+1) = isLinear(i);
					
				end				
			end
								
		end
		
		%	Update(e, curTrainMB, [curTrainLabelsMB])
		%	- performs a backpropogation weight update given the current training 
		%	minibatch 'curTrainMB' (rows: samples, col: dims) and the (optional, for 
		%	supervised learning) current training labels vector 'curTrainLabelsMB'
		function Update(obj, e, curTrainMB, varargin)
			
			if ~isempty(varargin)
				curTrainLabelsMB = varargin{1};
			end
			
			if ~obj.isClassifier % if DeepNN is not a classifier

				% vectorize weights/biases
				[wVect,wDims] = DeepNN.Weights2Vect(obj.weights, obj.biases);

				% use conjugate gradient minimization with backprop to
				% get updated weights/biases vector
				newWvect = minimize(wVect', 'GetError', ...
					obj.cgMaxSearches, wDims, obj.hidLinear, curTrainMB);

				% update weights/biases
				[newW, newB] = DeepNN.Vect2Weights(real(newWvect),wDims);
				obj.weights = newW;
				obj.biases = newB;

			else

				% if we're in the initial epochs, just update the top
				% level weights
				if e <= obj.numTopInitEpochs

					% prop cur mini batch up to penultimate layer
					[~,layerActivs] = obj.PropLayerActivs(curTrainMB, 1, ...
						length(obj.weights));
					topInputMB = layerActivs{end-1};

					% vectorize top level weights/biases
					[wVect,wDims] = DeepNN.Weights2Vect(obj.weights(end), ...
						obj.biases(end));

					% use conjugate gradient minimization with backprop to
					% get updated weights/biases vector
					newWvect = minimize(wVect', 'GetError', ...
						obj.cgMaxSearches, wDims, obj.hidLinear(end), topInputMB, ...
						curTrainLabelsMB);

					% update weights/biases
					[newW, newB] = DeepNN.Vect2Weights(real(newWvect),wDims);
					obj.weights(end) = newW;
					obj.biases(end) = newB;


				else

					% vectorize top level weights/biases
					[wVect,wDims] = DeepNN.Weights2Vect(obj.weights, ...
						obj.biases);

					% use conjugate gradient minimization with backprop to
					% get updated weights/biases vector
					newWvect = minimize(wVect', 'GetError', ...
						obj.cgMaxSearches, wDims, obj.hidLinear, curTrainMB, ...
						curTrainLabelsMB);

					% update weights/biases
					[newW, newB] = DeepNN.Vect2Weights(real(newWvect),wDims);
					obj.weights = newW;
					obj.biases = newB;

				end

			end
					
		end
		
		%	[trainSamples, valSamples] = GetSamples(trainData,valData,e)
		%	- gets the training and validation samples for the current epoch 'e' given 
		%	the variable (either matrix or cell array of external filenames) 
		%	'trainData' and'valData'
		function [trainSamples, valSamples] = GetSamples(obj,trainData,valData,e)
			
			% if the train/valData are just numeric matrices, assume we're in offline
			% mode
			if ~obj.trainOnline && isnumeric(trainData) && isnumeric(valData)
				trainSamples = trainData;
				valSamples = valData;
				
			% if the trainData is a cell array, we assume it's an ordered list of
			% files of data for online training
			elseif obj.trainOnline
				
				% if training data is given in cell array of filepaths
				if iscell(trainData)
					
					% assume that obj.online_numSampsPerSeg has been set, throw fatal error 
					% if it hasn't been set
					if isempty(obj.online_numSampsPerSeg)
						obj.MsgOut(['Error: obj.GetSamples, for online training ' ...
							'obj.online_numSampsPerSeg not set\n'],2);
						return
					end

					% calculate which segment file we want to load based on the epoch and
					% the number of samples per segment; assume that we'll use half the 
					% data for training and half for validation
					segFileInd = ceil(e*obj.miniBatchSize*2 / obj.online_numSampsPerSeg);

					% if segFileInd is larger than the list of files we have, throw fatal
					% error
					if segFileInd > length(trainData) 
						obj.MsgOut(sprintf(['Error: obj.GetSamples, trying to get samples ' ...
							'for epoch %d, but max number of samples is %d\n'], e, ...
							obj.online_numSampsPerSeg * length(trainData)),2);
						return
					end

					% load the appropriate segment file (slow to load possible the same
					% file at each function call, but it's currently cleaner than any sort
					% of caching we can do...hopefully Matlab will be smart about caching 
					% if possible)
					load(trainData{segFileInd}); % should load at least 'segSamples' var
				
				
					% get the samples for this epoch
					inds = mod((e-1)*obj.miniBatchSize*2 + (1:obj.miniBatchSize*2)-1, ...
						obj.online_numSampsPerSeg)+1;
					epSamples = segSamples(inds,:); %#ok<NODEF>

					% shuffle samples and split into train/val sets
					perm = randperm(obj.miniBatchSize*2);
					epSamples = epSamples(perm,:);
					trainSamples = epSamples(1:end/2,:);
					valSamples = epSamples((end/2+1):end,:);
					
				% if our data is numeric, just take a segment from it
				else 
					
					% get the samples for this epoch
					inds = (e-1)*obj.miniBatchSize + (1:obj.miniBatchSize);
					trainSamples = trainData(inds,:);
					valSamples = valData(inds,:);
					
				end
				
				
			% don't recognize training data format, so throw fatal error
			else
				obj.MsgOut('Error: obj.GetSamples, Unrecognized trainData format\n',2);
			end
				
			
		end
		
		%	LogEpoch(e, trainSamples, valSamples, [trainLabels], [valLabels])
		%	- calculate and log to output or file the epoch errors using the 
		%	'trainSamples' and 'valSamples' 2D matrices (and optionally, the 
		%	'trainLabels' and 'valLabels' vectors)
		function LogEpoch(obj,e, trainSamples, valSamples, varargin)
			
			% keep track of errors and timestamps
			obj.errorHistory(e).time = toc;
			
			if obj.isClassifier
				trainLabels = varargin{1};
				valLabels = varargin{2};
				
				% get train and val errors
				[trainCEE, trainRMSE, trainPercE] = obj.GetOnlyErrors(trainSamples, ...
					trainLabels);
				
				if ~isempty(valSamples)
					[valCEE, valRMSE, valPercE] = obj.GetOnlyErrors(valSamples, ...
						valLabels);
				else
					valCEE = -1;
					valRMSE = -1;
					valPercE = -1;
				end
				
				% store errors for this epoch
				obj.errorHistory(e).trainCEE = trainCEE;
				obj.errorHistory(e).trainRMSE = trainRMSE;
				obj.errorHistory(e).trainPercE = trainPercE;
				obj.errorHistory(e).valCEE = valCEE;
				obj.errorHistory(e).valRMSE = valRMSE;
				obj.errorHistory(e).valPercE = valPercE;
				
				% print out errors for this epoch
				obj.MsgOut(sprintf('%d\t%s\t\t%d\t%d\t%.04f\t\t%d\t%d\t%.04f\n',e, ...
						datestr(now), trainCEE, trainRMSE, trainPercE, valCEE, valRMSE, ...
						valPercE));
				
			else
				
				% get train and val errors
				[trainCEE, trainRMSE] = obj.GetOnlyErrors(trainSamples);
				
				if ~isempty(valSamples)
					[valCEE, valRMSE] = obj.GetOnlyErrors(valSamples);
				else
					valCEE = -1;
					valRMSE = -1;
				end
				
				% store errors for this epoch
				obj.errorHistory(e).trainCEE = trainCEE;
				obj.errorHistory(e).trainRMSE = trainRMSE;
				obj.errorHistory(e).valCEE = valCEE;
				obj.errorHistory(e).valRMSE = valRMSE;
				
				% print out errors for this epoch
				obj.MsgOut(sprintf('%d\t%s\t%d\t%d\t%d\t%d\n',e,datestr(now), ...
						trainCEE, trainRMSE, valCEE, valRMSE));
			end

			
			% store current weights
			obj.weightsHistory{e} = obj.weights;
			obj.biasesHistory{e} = obj.biases;
			
			% delete previous weights/biases history if necessary
			earlyE = e - obj.overfitHistory - 1;
			if earlyE > 1 && mod(earlyE,obj.historyLogFreq) ~= 0
				obj.weightsHistory{earlyE} = {};
				obj.biasesHistory{earlyE} = {};
			end
			
		end
		
		%	MakeClassifier(outDim)
		%	- removes the decoding layers of the DeepNN and inits the classification
		%	layer and weights
		function MakeClassifier(obj, outDim)
			
			obj.isClassifier = 1;
			
			% remove decoding weights
			obj.weights = obj.weights(1:end/2);
			obj.biases = obj.biases(1:end/2);
			%obj.hidLinear = obj.hidLinear(1:end/2);
			obj.hidLinear = zeros(size(obj.hidLinear(1:end/2)));
			
			% add logistic regression layer with random initialization
			%initSpread = 4*sqrt(6/(length(obj.biases{end}) + outDim));
			%obj.weights{end+1} = -initSpread + 2*initSpread*rand(length(obj.biases{end}), ...
			%	outDim);
			obj.weights{end+1} = 0.1*randn(length(obj.biases{end}),outDim);
			obj.biases{end+1} = 0.1*randn(1,outDim);
			obj.hidLinear(end+1) = 1;

			obj.errorHistory = [];
			obj.weightsHistory = [];
			obj.biasesHistory = [];
			
		end
		
		%	[CEE, RMSE, [PercMissed]] = GetOnlyErrors(input, [target])
		%	- calculates the cross-entropy error ('CEE') and root-mean-squared error
		%	('RMSE') as well as the misclassification error 'PercMissed' for 
		%	classifiers based on the 'input' data (rows: samples, cols: dims), and the
		%	optional 'target' labels
		function [CEE, RMSE, PercMissed] = GetOnlyErrors(obj, input, varargin)
			
			if length(varargin) >= 1 % if we're doing classification
				target = varargin{1};
				[~,targetLabels] = max(target,[],2);
			end
			
			% get output for error calculations
			MBOut = obj.PropLayerActivs(input);
			
			% avoid NaNs
			logVals1 =  log(MBOut);
			logVals1(isnan(logVals1)) = 0;
			logVals2 =  log(1-MBOut);
			logVals2(isnan(logVals2)) = 0;
			
			% do error calcs
			if length(varargin) >= 1 % if it's a classifier
				CEE = -mean(sum(target .* logVals1));
				RMSE = mean(sqrt(mean((target - MBOut).^2)));
				
				[~,MBOutLabels] = max(MBOut,[],2);
				PercMissed = sum(MBOutLabels ~= targetLabels) * 100 / ...
					length(targetLabels); 
			else
				CEE = -mean(sum(input .* logVals1 + (1-input) .* logVals2));
				RMSE = mean(sqrt(mean((input - MBOut).^2)));
				PercMissed = [];
			end
						
		end
		
		%	isOverfit = CheckOverfit()
		%	- uses validation error to check whether the current model is overfit,
		%	returning a logical 'isOverfit' value
		function isOverfit = CheckOverfit(obj)
			
			% use the RMSE errors as the metric of overfit
			recentErrors = [obj.errorHistory(end-obj.overfitHistory:end).valRMSE];
			
			% determine if model is overfit if the earliest error is the smallest of
			% the other recent errors
			[~,epochInds] = sort(recentErrors,'ascend');
			isOverfit = epochInds(1) == 1;
			
			if isOverfit
				
				% revert model to best (earliest) model in recent history
				obj.weights = obj.weightsHistory{end-obj.overfitHistory};
				obj.biases = obj.biasesHistory{end-obj.overfitHistory};
				
				% output message to this effect
				obj.MsgOut(sprintf('Model is overfit; reverting to model at epoch %d\n', ...
					length(obj.weightsHistory) - obj.overfitHistory));
				
			end
			
		end
		
		%	dnnObj = AttemptLoadModel()
		% - checks to see if a model file at the current obj.modelFile path exists,
		%	and if it does loads it and sets the obj to the model
		function dnnObj = AttemptLoadModel(obj) 
			
			dnnObj = obj;
			
			if obj.attemptLoad && exist(obj.modelFile,'file')
				try
					load(obj.modelFile);
					[~,filename] = fileparts(obj.modelFile);
					obj.MsgOut(sprintf('model file %s.mat loaded successfully\n',filename));
				catch ex
					obj.MsgOut(sprintf('model file %s.mat load error: %s\n',filename, ...
						ex.message));
				end				
			end
						
		end
		
	end
    
	% PRIVATE METHODS -----------------------------------------------------
	methods (Access = private)
	end

    
end
