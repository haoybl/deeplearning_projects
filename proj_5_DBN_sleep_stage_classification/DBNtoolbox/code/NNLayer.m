classdef NNLayer < handle 
%
% NNLayer is an abstract class that contains properties and method
% implementations common to single hidden-layer neural networks (e.g. RBM)
%
% CONSTRUCTORS
%   
%   obj = NNLayer(theNumVisibleUnits, theNumHiddenUnits, [params])
%		- 'theNumVisibleUnits' and 'theNumHiddenUnits' are scalars giving the number
%		of visible and hidden units; 'params' is an optional struct with fields
%		and values of the setable model parameters
%
%
% PUBLIC PROPERTIES [Public Read-Only, Protected Read/Write]
%
%		numVis :				the number of visible units
%
%		numHid :				the number of hidden units
%
%   weights :				the undirected weights between the two layers in the form of
%										a 2D matrix with 'numVisUnits' rows and 'numHidUnits' cols
%
%		hidBiases :			a 1-by-'numHidBiases' vector of biases to the hidden layer
%
%		visBiases :			a 1-by-'numVisBiases' vector of biases to the visible layer
%
%		hidLinear :			a scalar (0/1) describing whether the hidden units are
%										linear (default: 0)
%
%		visLinear :			a scalar (0/1) describing whether the visible units are
%										linear (default: 0)
%
%		errorHistory :	a struct array with an element for each training epoch
%										with fields,
%											.time : the number of seconds after the start of training
%											.trainRMSE : the training root mean-squared-error
%											.trainCEE : the training cross-entropy error
%											.valRMSE : the validation (if supplied) root 
%												mean-squared-error 
%											.valCEE : the validation (if supplied) cross-entropy error
%
%		weightsHistory :	a struct array with the weights/biases every
%											obj.weightsLogFreq epochs
%
%		params :				a struct with the parameters of the model
%
%
% PUBLIC METHODS
%
%		Train(obj, trainSamples, [valSamples], [isHidLinear], [isVisLinear])
%		- Train uses the MxD 'trainSamples' matrix (M samples with D dimensions)
%		to train the NNL via the subclass's 'Update' function; 'valSamples' is M2xD
%		(where M2 is not necessarily the same as M); 'isHidLinear' and 'isVisLinear'
%		(0/1) optionally define whether the hidden and visible layers are linear
%
%		[weights, hidBiases, visBiases] = GetModel(obj)
%		- GetModel returns the weights and biases of the model
%
%		SetModel(obj, weights, hidBiases, visBiases, hidLinear, visLinear)
%		- SetModel sets the parameters of the model
%
%
%		ABSTRACT METHODS
%		----------------------------------------------------------------------------
%
%		hidActivs = PropHidFromVis(obj, visActivs) 
%		- PropHidFromVis propogates a matrix of visible layer activations up to get 
%		hidden layer activations
%
%		visActivs = PropVisFromHid(obj, hidActivs) 
%		- PropVisFromHid propogates a matrix of hidden layer activations down to get
%		visible layer activations
%
%		reconSamples = GetRecon(obj, samples)
%		- GetRecon reconstructs the sample according to the specific subclass
%		implementation
%		
%
%		STATIC METHODS
%		----------------------------------------------------------------------------
%
%		cee = GetCEE(orig, recon)
%		- GetCEE computes the cross-entropy error between the original and
%		reconstructed samples
%
%		rmse = GetRMSE(orig, recon)
%		- GetRMSE computes the mean squared error between the original and 
%		reconstruction over the dimensions and samples
%
%		y = Sigm(x) 
%		- Sigm defines the logistic sigmoid function, y(x) = 1/(1 + exp(-x))
%
%
% EXTERNAL CLASS AND FILE REQUIREMENTS
%
%   None 
%
%
% KNOWN INCOMPATIBILITIES 
%
%		None
%
%
% ACKNOWLEDGEMENTS 
%
%   This code is taken in part from the rbm.m script written by Ruslan 
%		Salakhutdinov and Geoff Hinton, whose complete code can be found 
%   at http://www.cs.toronto.edu/~hinton/MatlabForSciencePaper.html as
%   well as the RBM class from the Theano documenation,
%   http://www.deeplearning.net/tutorial/rbm.html
%
%
%	Created by:
%   Drausin Wulsin
%   21/07/2010
%  	Translational Neuroengineering Lab, University of Pennsylvania
%
%
%
      
    
	% PUBLIC PROPERTIES ---------------------------------------------------
	properties (GetAccess = public, SetAccess = protected)

		% the number of visible units
		numVis

		% the number of hidden units
		numHid
		
		% the undirected weights between the two layers in the form of a 2D matrix 
		% with 'numVisUnits' rows and 'numHidUnits' cols
		weights

		% a 1x'numHidBiases' vector of biases to the hidden layer
		hidBiases

		% a 1x'numVisBiases' vector of biases to the visible layer
		visBiases
		
		% a scalar (0/1) describing whether the hidden units are linear
		hidLinear = 0
		
		% a scalar (0/1) describing whether the visible units are linear
		visLinear = 0
		
		% a struct array with an element for each training epoch
		%	with fields,
		%	.time : the number of seconds after the start of training
		%	.trainRMSE : the training root mean-squared-error
		%	.trainCEE : the training cross-entropy error
		%	.valRMSE : the validation (if supplied) root mean-squared-error 
		%	.valCEE : the validation (if supplied) cross-entropy error
		errorHistory
		
		% a struct array with the weights/biases every obj.weightsLogFreq epochs
		weightsHistory
		
	end
	
	properties (GetAccess = public, Abstract, Dependent)
		
		% a struct with the parameters of the model
		params
				
	end
    
	% PROTECTED PROPERTIES ------------------------------------------------
	properties (Access = protected)
		
		% the weight update values from the previous epoch
		weightDeltas
		
		% the hidden bias update values from the previous	epoch
		hidBiasDeltas
		
		% the visible bias update values from the previous	epoch
		visBiasDeltas
		
		% the best model (using training or validation RMSE) over the	epochs, used 
		% to revert to best model at end of training if	necessary
		bestModel
		
		% SETABLE PARAMETERS: set any of these parameters by setting a field with
		% their same name in the 'params' struct, which is passed either into the
		% constructor or to the obj.SetParams(params) function
		
		% the string ID of the object, used as prefix for model and status files
		strID 
		
		% the number of training epochs
		numEpochs = 50;
		
		% a vector (length >= obj.numEpochs) of the learning rate for each epoch 
		learningRate = 0.05*ones(1,1000);
		
		% the number of samples to combine into one	mini-batch
		miniBatchSize = 100;
		
		% a vector (length >= obj.numEpochs) of momentum values for each epoch that 
		% control how much of the previous epoch's weight/bias deltas determine the 
		% current epoch
		momentum = [0.5*ones(1,5) 0.9*ones(1,1000-5)];
		
		% a vector (length >= obj.numEpochs) of the weight	costs, which determine 
		% how much the previous epoch's weights influence the current weight update
		weightCost = 0.0002*ones(1,1000);
		
		% a string, either 'uniform' or 'normal', that determines the distribution 
		% of the initialized weights and biases
		initDist = 'uniform';
		
		% the spead of the weight/biases initialization, if obj.initDist == 
		% 'normal', then 'initSpread' is the standard deviation of the weights with 
		% zero mean; if obj.initDist == 'uniform', then 'initSpread' is	the distance 
		% on either side of zero, that the	distribution extends to
		initSpread 
		
		% a scalar (0/1) whether to calculate validation error
		calcValError = 1;
		
		% the frequency of weight/bias storage
		weightsLogFreq = 0;
		
		% the verbosity level during training:
		%	0 - no epoch output
		%	1 - epoch error written to console and NNL saved to file
		%	2 - epoch error and NNL written to files
		verbosity = 2;
		
		% the directory where output files are placed
		outDir = '';
		
		% a scalar (0/1) whether to look for and load a model file with the same 
		% model-file name
		attemptLoad = 1;
		
		
		% OTHER PROPERTIES (no setable)
		
		% a flag indicating a fatal error, prevents training from starting if true
		fatalFlag = 0;
	
		% the file where the epoch error info is written if verbosity = 2
		statusFile
	
		% the file where this object (obj) is written if verbosity >= 1
		modelFile
		
	end
    
	% PRIVATE PROPERTIES --------------------------------------------------
	properties (Access = private)
	end


	% PUBLIC METHODS ------------------------------------------------------
	methods (Access = public)

		% CONSTRUCTOR
		%
		% obj = NNLayer(theNumVisibleUnits, theNumHiddenUnits, [params])
		%	- 'theNumVisibleUnits' and 'theNumHiddenUnits' are scalars giving the 
		%	number of visible and hidden units; 'params' is an optional struct with 
		%	fields and values of the setable model parameters
		function obj = NNLayer(theNumVisibleUnits, theNumHiddenUnits, varargin)

			% VALIDATE ARGUMENTS ------------------------------------------

			% theNumVisibleUnits should be a positive integer
			validateattributes(theNumVisibleUnits,{'double'},{'positive'});

			% theNumHiddenUnits should be a positive integer
			validateattributes(theNumHiddenUnits,{'double'},{'positive'});

			% params should be a struct
			if nargin == 3
				params = varargin{1};
				validateattributes(params,{'struct'},{});
			end

			% END VALIDATE ARGUMENTS --------------------------------------

			% set number of units
			obj.numVis = theNumVisibleUnits;
			obj.numHid = theNumHiddenUnits;

		end

		%	Train(obj, trainSamples, [valSamples], [isHidLinear], [isVisLinear])
		%	- Train uses the MxD 'trainSamples' matrix (M samples with D dimensions)
		%	to train the NNL via the subclass's 'Update' function; 'valSamples' is 
		%	M2xD (where M2 is not necessarily the same as M); 'isHidLinear' and 
		%	'isVisLinear' (0/1) optionally define whether the hidden and visible 
		% layers are linear
		function Train(obj, trainSamples, valSamples, varargin)
					
			% determine whether the hidden and visible layers are linear or not
			if length(varargin) >= 1
				obj.hidLinear = varargin{1};
			end
			if length(varargin) >= 2
				obj.visLinear = varargin{2};
			end
			
			% determine if valSamples have actually been passed in
			if ~exist('valSamples','var') || isempty(valSamples)
				valSamples = [];
				obj.calcValError = 0;
			else
				obj.calcValError = 1;
			end
			
			% check for fatal flag
			if obj.fatalFlag
				obj.MsgOut('Fatal Error: training aborted\n',2);
				return
			end
						
			% calc number of mini-batches to use
			numMBs = floor(size(trainSamples,1) / obj.miniBatchSize);
			
			% start log and timer
			if ~obj.calcValError
				obj.MsgOut(['Epoch\tDate-Timestamp\tAvgTrainCEE\t' ...
					'AvgTrainRMSE\n']);
			else
				obj.MsgOut(['Epoch\tDate-Timestamp\tAvgTrainCEE\t' ...
				'AvgTrainRMSE\t\tAvgValCEE\tAvgValRMSE\n']);
			end
			tic;
			
			% epochs loop, starting either at the beginning or where last training
			% left off
			for e = (length(obj.errorHistory)+1):obj.numEpochs
				
				for mb = 1:numMBs
					
					% get the current mini-batch
					curMB = trainSamples((mb-1)*obj.miniBatchSize + ...
						(1:obj.miniBatchSize),:);
					
					% upate the params 
					obj.Update(curMB,e);
					
				end
				
				% log epoch info
				obj.LogEpoch(e, trainSamples, valSamples);
				
				% check for early stoping condition
				if obj.CheckEarlyStop(e)
					break
				end
				
			end
			
			% reset to best model, if necessary
			if obj.bestModel.atEpoch ~= e
				obj.MsgOut(sprintf('Resetting to best model at epoch %d\n', ...
					obj.bestModel.atEpoch));
				obj.weights = obj.bestModel.weights;
				obj.visBiases = obj.bestModel.visBiases;
				obj.hidBiases = obj.bestModel.hidBiases;
			end
			
		end
		
		%	[weights, hidBiases, visBiases] = GetModel()
		%	- GetModel returns the weights and biases of the model
		function [weights, hidBiases, visBiases] = GetModel(obj)
			weights = obj.weights;
			hidBiases = obj.hidBiases;
			visBiases = obj.visBiases;
		end
		
		%	SetModel(obj, weights, hidBiases, visBiases, hidLinear, visLinear)
		%	- SetModel sets the parameters of the model
		function SetModel(obj, weights, hidBiases, visBiases, hidLinear, visLinear)
			
			obj.weights = weights;
			obj.hidBiases = hidBiases;
			obj.visBiases = visBiases;
			obj.hidLinear = hidLinear;
			obj.visLinear = visLinear;
			
			obj.numHid = length(hidBiases);
			obj.numVis = length(visBiases);
			
		end
		
	end
	
	methods (Access = public, Abstract)
		
		%	hidActivs = PropHidFromVis(obj, visActivs) 
		%	- PropHidFromVis propogates a matrix of visible layer activations up to 
		%	get hidden layer activations
		hidActivs = PropHidFromVis(obj, visActivs)
			
		%	visActivs = PropVisFromHid(obj, hidActivs) 
		%	- PropVisFromHid propogates a matrix of hidden layer activations down to 
		%	get visible layer activations
		hidActivs = PropVisFromHid(obj, hidActivs)
		
		% reconSamples = GetRecon(obj, samples)
		%	- GetRecon reconstructs the sample according to the specific subclass
		%	implementation
		reconSamples = GetRecon(obj, samples)
		
	end
	
	methods (Access = public, Static)
		
		%	cee = GetCEE(orig, recon)
		%	- GetCEE computes the cross-entropy error between the original and
		%	reconstructed samples
		function cee = GetCEE(orig, recon)
			
			logVals1 = log(recon);
			logVals2 = log(1-recon);
			
			% avoid NaNs
			logVals1(isnan(logVals1)) = 0;
			logVals2(isnan(logVals2)) = 0;
			
			cee = -mean(sum(orig .* logVals1 + (1-orig) .* logVals2,2));
		end
		
		%	rmse = GetRMSE(orig, recon)
		%	- GetRMSE computes the mean squared error between the original and 
		%	reconstruction over the dimensions and samples
		function rmse = GetRMSE(orig, recon)
			rmse = mean(sqrt(mean((orig - recon).^2)));
		end
		
		% y = Sigm(x) 
		%	- Sigm defines the logistic sigmoid function, y(x) = 1/(1 + exp(-x))
		function y = Sigm(x)
			y = 1 ./ (1 + exp(-x));
		end
		
	end
	  
    % PROTECTED METHODS ---------------------------------------------------
    methods (Access = protected)
		
		% MsgOut(text, isError)
		%	- MsgOut outputs string in 'text' to either console or a file, depending  
		%	on verbosity properties; 'isError' is optional but if its value is > 0,  
		%	'text' will be printed to console even with verbosity 0; if 'isError' 
		%	value is > 1, 'obj.fatalFlag' is set to 1
		function MsgOut(obj, text, isError)
			
			% set isError to 0 if it doesn't exist
			if ~exist('isError','var')
				isError = 0;
			end
			
			if obj.verbosity == 1 || isError
				fprintf(text)
			end
			
			if obj.verbosity == 2 
				fid = fopen(obj.statusFile,'a');
				fprintf(fid,text);
				fclose(fid);
			end
			
			% set fatalFlag if necessary
			if isError > 1
				obj.fatalFlag = 1;
			end
			
		end
		
		%	Init()
		%	- Init initializes the model weights and biases
		function Init(obj)
						
			% init weights
			if strcmp(obj.initDist,'normal')
				
				obj.weights = obj.initSpread * randn(obj.numVis,obj.numHid);
				
			else
				if ~strcmp(obj.initDist,'uniform')
					obj.MsgOut(sprintf(['Invalid weight distribution: %s, ' ...
						'defaulting to uniform distribution\n'],obj.initDist),1);
				end
				
				obj.weights = -obj.initSpread + 2*obj.initSpread * ...
					rand(obj.numVis,obj.numHid);
			end
			
			% init biases
			obj.visBiases = zeros(1,obj.numVis);
			obj.hidBiases = zeros(1,obj.numHid);
			
			% init changes (deltas) for weights/biases to zero
			obj.weightDeltas = zeros(obj.numVis,obj.numHid);
			obj.visBiasDeltas = zeros(1,obj.numVis);
			obj.hidBiasDeltas = zeros(1,obj.numHid);
			
			% init bestModel error
			obj.bestModel.rmse = Inf;
						
		end
		
		%	LogEpoch(e, trainSamples, valSamples)
		%	- LogEpoch calculates the training (and validation) errors, logs the
		%	weights/biases, saves the model file, and write the status file output; it
		%	also calls the 'AtEpoch' function
		function LogEpoch(obj, e, trainSamples, valSamples)
			
			% keep track of errors and timestamps
			obj.errorHistory(e).time = toc;

			trainRecon = obj.GetRecon(trainSamples);
			obj.errorHistory(e).trainCEE = obj.GetCEE(trainSamples,trainRecon);
			obj.errorHistory(e).trainRMSE = obj.GetRMSE(trainSamples,trainRecon);

			if obj.calcValError
				valRecon = obj.GetRecon(valSamples);
				obj.errorHistory(e).valCEE = obj.GetCEE(valSamples,valRecon);
				obj.errorHistory(e).valRMSE = obj.GetRMSE(valSamples,valRecon);
			end

			% log epoch
			if ~obj.calcValError
				obj.MsgOut(sprintf('%d\t%s\t%d\t%d\n',e,datestr(now, ...
					'mm/dd HH:MM:SS'),obj.errorHistory(e).trainCEE, ...
					obj.errorHistory(e).trainRMSE));
			else
				obj.MsgOut(sprintf('%d\t%s\t%d\t%d\t\t%d\t%d\n',e,datestr(now, ...
					'mm/dd HH:MM:SS'),obj.errorHistory(e).trainCEE, ...
					obj.errorHistory(e).trainRMSE, obj.errorHistory(e).valCEE, ...
					obj.errorHistory(e).valRMSE));
			end

			% log weights and biases
			if e == 1 || mod(e,obj.weightsLogFreq) == 0
				obj.weightsHistory(e).weights = obj.weights;
				obj.weightsHistory(e).visBiases = obj.visBiases;
				obj.weightsHistory(e).hidBiases = obj.hidBiases;
			end
			
			% log update bestModel, if current is best
			if obj.calcValError && obj.errorHistory(e).valRMSE < obj.bestModel.rmse
					obj.bestModel.rmse = obj.errorHistory(e).valRMSE;
					obj.bestModel.weights = obj.weights;
					obj.bestModel.visBiases = obj.visBiases;
					obj.bestModel.hidBiases = obj.hidBiases;
					obj.bestModel.atEpoch = e;					
			elseif obj.errorHistory(e).trainRMSE < obj.bestModel.rmse
					obj.bestModel.rmse = obj.errorHistory(e).trainRMSE;
					obj.bestModel.weights = obj.weights;
					obj.bestModel.visBiases = obj.visBiases;
					obj.bestModel.hidBiases = obj.hidBiases;
					obj.bestModel.atEpoch = e;					
			end

			% save this object
			nnlObj = obj; %#ok<NASGU>
			save(obj.modelFile,'nnlObj');

			% call AtEpoch function, which may also do nothing (specified in
			% sub-class)
			obj.AtEpoch(trainSamples);
			
		end
		
		%	AtEpoch(varargin)
		%	- AtEpoch is called after each epoch and contains any code to be executed
		%	after each epoch (debugging or otherwise)
		function AtEpoch(obj,varargin) %#ok<MANU>
		end
		
		%	[nnlObj, success] = AttemptLoadModel() 
		%	- AttemptLoadModel check if model file already exists and loads it if
		%	obj.attemptLoad == 1 and if the file exists
		function [nnlObj, success] = AttemptLoadModel(obj) 
			
			nnlObj = obj;
			success = 0;
			
			if obj.attemptLoad && exist(obj.modelFile,'file')
				try
					load(obj.modelFile);
					[~,filename] = fileparts(obj.modelFile);
					success = 1;
					obj.MsgOut(sprintf('model file %s.mat loaded successfully\n',filename));
				catch ex
					obj.MsgOut(sprintf('model file %s.mat load error: %s\n',filename, ...
						ex.message));
				end				
			end
						
		end
		
		%	stop = CheckEarlyStop(curEpoch)
		%	- CheckEarlyStop checks the training (and validation) error to see if it 
		%	is worse than that of the first epoch and thus training needs to stop
		function stop = CheckEarlyStop(obj,curEpoch)
			
			stop = 0;
			
			% check to see if either trainRMSE or valRMSE is larger than that of first
			% epoch
			if curEpoch > 20 
				if obj.errorHistory(curEpoch).trainRMSE > obj.errorHistory(1).trainRMSE
					obj.MsgOut('Train error worse than 1st epoch, stopping\n');
					stop = 1;
				elseif isfield(obj.errorHistory(curEpoch),'valRMSE') && ...
						obj.errorHistory(curEpoch).valRMSE > obj.errorHistory(1).valRMSE
					obj.MsgOut('Val error worse than 1st epoch, stopping\n');
					stop = 1;
				end
			end
		end
		
	end
	
	methods (Access = protected, Abstract)
		
		%	Update(visSamples_0, e) 
		%	- UpdateModel updates the weights and biases of the model for inputs
		%	visSamples and returns the cross-entropy and mean-squared error
		[cee,rmse] = Update(obj, visSamples_0, e)
		 
	end
    
    % PRIVATE METHODS -----------------------------------------------------
    methods (Access = private)
    end
    
    
end
