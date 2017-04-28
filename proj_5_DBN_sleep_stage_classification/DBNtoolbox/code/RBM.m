classdef RBM < NNLayer
%
% RBM implements a basic Restricted Boltzman Machine trained with 
% contrastive divergance and inherits many properties/methods from the 
% abstract NNLayer class.
%
% CONSTRUCTORS
%   
%		obj = RBM(theNumVisibleUnits, theNumHiddenUnits, [params])
%		- 'theNumVisibleUnits' and 'theNumHiddenUnits' are scalars giving the number
%		of visible and hidden units; 'params' is an optional struct with fields
%		and values of the setable model parameters
%
%
% PUBLIC PROPERTIES [Public Read-Only, Protected Read/Write]
%
%		params :			a struct containing parameter values for the RBM 
%
%
% PUBLIC METHODS
%
%		[visActivs_k, hidActivs_0] = GibbsSample(visSamples_0, k)
%		- GibbsSample performs 'k' iterations of Gibbs sampling from 'visSamples_0',
%		an MxD matrix of M samples with D dimensions, and returns MxD matrix 
%		'visActivs_k', the activations after 'k' Gibbs sampling iterations, and 
%		MxF matrix 'hidActivs_0', where F is the dimensionality of the hidden
%		layer
%
%		[hidSample,hidActivs] = SampleHidGivenVis(visActivs)
%		- calculates the hidden activations 'hidActivs' (MxF) from the visible 
%		activations 'visActivs' (MxD reals) and samples from the hidden units w/ 
%		probs defined in 'hidActivs' to get 'hidSamples'
%
%		[visSample,visActivs] = SampleVisGivenHid(hidActivs)
%		- calculates the visible activations 'visActivs' (MxD) from the
%		hidden activations 'hidActivs' (MxF) and samples from the visible
%		units w/ probs defined in 'visActivs' to get 'visSamples' 
%
%		hidActivs = PropHidFromVis(visActivs)
%		- propogates the visible activations 'visActivs' (MxD) to the hidden layer
%		to get the hidden activations 'hidActivs' (MxF)
%
%		visActivs = PropVisFromHid(hidActivs)
%		- propogates the hidden activations 'hidActivs' (MxF) to the hidden layer
%		to get the visible activations 'visActivs' (MxD)
%
%		reconSamples = GetRecon(samples)
%		- uses Gibbs sampling to get reconstruction 'reconSamples' (MxD) from the
%		original 'samples' (MxD)
%
%		[STATIC] [visActivs_k, hidActivs_0, hidSamples_n] = GibbsSample_S(weights, ...
%			hidBiases, visBiases, hidLinear, visLinear, visSamples_0, [k])
%		- static version of GibbsSample with 'weights' (DxF) between the layers,
%		'hidBiases' (1xF), 'visBiases' (1xD), and logical 'hidLinear' and
%		'visLinear' scalars
%
%		[STATIC] [hidSample,hidActivs] = SampleHidGivenVis_S(weights, hidBiases, ...
%			hidLinear, visActivs)
%		- static version of SampleHidGivenVis with 'weights' (DxF) between the 
%		layers, 'hidBiases' (1xF), and logical 'hidLinear' scalar
%
%		[STATIC] [visSample,visActivs] = SampleHidGivenVis_S(weights, hidBiases, ...
%			hidLinear, visActivs)
%		- static version of SampleHidGivenVis with 'weights' (DxF) between the 
%		layers, 'visBiases' (1xD), and logical 'visLinear' scalar
%
%		[STATIC] hidActivs = PropHidFromVis_S(weights, hidBiases, hidLinear, ...
%			visActivs)
%		- static version of PropHidGivenVis with 'weights' (DxF) between the 
%		layers, 'hidBiases' (1xF), and logical 'hidLinear' scalar
%
%		[STATIC] visActivs = PropVisFromHid_S(weights, visBiases, visLinear, ...
%			hidActivs)
%		- static version of PropHidGivenVis with 'weights' (DxF) between the 
%		layers, 'visBiases' (1xD), and logical 'visLinear' scalar
%
%
% EXTERNAL CLASS AND FILE REQUIREMENTS
%
%   NNLayer.m 
%
%
% KNOWN INCOMPATIBILITIES 
%
%		None
%
%
% ACKNOWLEDGEMENTS 
%
%   This code is heavily adapted from the rbm.m script written by Ruslan 
%		Salakhutdinov and Geoff Hinton, whose complete code can be found 
%   at http://www.cs.toronto.edu/~hinton/MatlabForSciencePaper.html as
%   well as the RBM class from the Theano documenation,
%   http://www.deeplearning.net/tutorial/rbm.html
%
%
% CONTRIBUTORS 
%
%	Created by:
%   	Drausin Wulsin
%   	21/07/2010
%			Translational Neuroengineering Lab, University of Pennsylvania
%
%
%
%
% TODO:
% - add pseudo-likelihood cost calc
%

	% PUBLIC PROPERTIES ---------------------------------------------------
	properties (Access = public)
	end
	
	properties (GetAccess = public, Dependent)
		
		params
				
  end
    
	% PROTECTED PROPERTIES ------------------------------------------------
	properties (Access = protected)
		
		% the number of Gibbs sampling iterations to do in contrastive divergence		
		numCDIters = 1;
				
  end
    
	% PRIVATE PROPERTIES --------------------------------------------------
	properties (Access = private)
	end
    
    
	% PUBLIC METHODS ------------------------------------------------------
	methods (Access = public)
        
		% CONSTRUCTOR
		%
		%	obj = RBM(theNumVisibleUnits, theNumHiddenUnits, [params])
		%	- 'theNumVisibleUnits' and 'theNumHiddenUnits' are scalars giving the 
		%	number of visible and hidden units; 'params' is an optional struct with 
		%	fields and values of the setable model parameters
		function obj = RBM(theNumVisibleUnits, theNumHiddenUnits, varargin)
						
			% send arguments to the NNLayer constructor
			obj = obj@NNLayer(theNumVisibleUnits, theNumHiddenUnits, varargin{:});
			
			% set internal params
			obj.SetInternalParams();
			
			% set any parameters passed in
			if nargin == 3
				obj.SetParams(varargin{1});
			else
				obj.SetParams(struct); % pass empty struct
			end
						
			% initialize model properties
			obj.Init();
			
			% load the model file if it already exists
			obj = obj.AttemptLoadModel();
						
		end
		
		%	[visActivs_k, hidActivs_0] = GibbsSample(visSamples_0, [k])
		%	- performs 'k' (default: 1) iterations of Gibbs sampling from 'visSamples_0',
		%	an MxD matrix of M samples with D dimensions, and returns MxD matrix 
		% 'visActivs_k', the activations after 'k' Gibbs sampling iterations, and 
		% MxF matrix 'hidActivs_0', where F is the dimensionality of the hidden
		% layer
		function [visActivs_k, hidActivs_0] = GibbsSample(obj,visSamples_0, k)
			
			% default k = 1
			if ~exist('k','var')
				k = 1;
			end
			
			% current visSample 
			visActivs_n = visSamples_0;
			
			% iterate
			for i = 1:k
				
				% hidden values as stochastic binary states
				[hidSamples_n, hidActivs_n] = obj.SampleHidGivenVis(visActivs_n);
				visActivs_n = obj.PropVisFromHid(hidSamples_n);
				
				%hist(obj.weights(:),50); drawnow;
				
				%hidActivs_n = obj.PropHidFromVis(visActivs_n);
				%visActivs_n = obj.PropVisFromHid(hidActivs_n);
				
				if i == 1
					hidActivs_0 = hidActivs_n;
				end
			end
			
			% set kth sample to last one in chain
			visActivs_k = visActivs_n;
			
		end
		
		% [hidSample,hidActivs] = SampleHidGivenVis(visActivs)
		% - calculates the hidden activations 'hidActivs' (MxF) from the visible 
		% activations 'visActivs' (MxD reals) and samples from the hidden units w/ 
		% probs defined in 'hidActivs' to get 'hidSamples'
		function [hidSamples,hidActivs] = SampleHidGivenVis(obj, visActivs)
			hidActivs = obj.PropHidFromVis(visActivs);
			
			% if the units are [0 1]-valued units
			if ~obj.hidLinear
				hidSamples = double(rand(size(hidActivs)) < hidActivs);
			
			else % if the units are linear units, assume unit variance
				hidSamples = hidActivs + randn(size(hidActivs));
			end
		end
		
		% [visSample,visActivs] = SampleVisGivenHid(hidActivs)
		% - calculates the visible activations 'visActivs' (MxD) from the
		% hidden activations 'hidActivs' (MxF) and samples from the visible
		% units w/ probs defined in 'visActivs' to get 'visSamples' 
		function [visSamples,visActivs] = SampleVisGivenHid(obj, hidActivs)
			visActivs = obj.PropVisFromHid(hidActivs);
			
			% if the units are [0 1]-valued units
			if ~obj.visLinear
				visSamples = double(rand(size(visActivs)) < visActivs);
				
			else  % if the units are linear units, assume unit variance
				visSamples = visActivs + randn(size(hidActivs));
			end
			
		end
		
		% hidActivs = PropHidFromVis(visActivs)
		% - propogates the visible activations 'visActivs' (MxD) to the hidden layer
		% to get the hidden activations 'hidActivs' (MxF)
		function hidActivs = PropHidFromVis(obj, visActivs)
			
			% if the units are [0 1]-valued units
			if ~obj.hidLinear
				hidActivs = obj.Sigm(visActivs * obj.weights + ...
					repmat(obj.hidBiases,size(visActivs,1),1));
			else % if the units are linear units
				hidActivs = visActivs * obj.weights + ...
					repmat(obj.hidBiases,size(visActivs,1),1);
			end
			
		end
		
		% visActivs = PropVisFromHid(hidActivs)
		% - propogates the hidden activations 'hidActivs' (MxF) to the hidden layer
		% to get the visible activations 'visActivs' (MxD)
		function visActivs = PropVisFromHid(obj, hidActivs)
			
			if ~obj.visLinear % if the units are [0 1]-valued units
				visActivs = obj.Sigm(hidActivs * obj.weights' + ...
					repmat(obj.visBiases,size(hidActivs,1),1));
			
			else % if the units are linear units
				visActivs = hidActivs * obj.weights' + ...
					repmat(obj.visBiases,size(hidActivs,1),1);
			end
		end
		
		% reconSamples = GetRecon(samples)
		% - uses Gibbs sampling to get reconstruction 'reconSamples' (MxD) from the
		% original 'samples' (MxD)
		function reconSamples = GetRecon(obj, samples)
			reconSamples = obj.GibbsSample(samples,obj.numCDIters);
		end
		
	end	
	
	methods (Access = public, Static)
		
		% [visActivs_k, hidActivs_0, hidSamples_n] = GibbsSample_S(weights, ...
		%		hidBiases, visBiases, hidLinear, visLinear, visSamples_0, [k])
		% - static version of GibbsSample with 'weights' (DxF) between the layers,
		% 'hidBiases' (1xF), 'visBiases' (1xD), and logical 'hidLinear' and
		% 'visLinear' scalars
		function [visActivs_k, hidActivs_0, hidSamples_n] = GibbsSample_S(weights, ...
				hidBiases, visBiases, hidLinear, visLinear, visSamples_0, k)
			
			% default k = 1
			if ~exist('k','var')
				k = 1;
			end
			
			% current visSample 
			visActivs_n = visSamples_0;
			
			% iterate
			for i = 1:k
				
				% hidden values as stochastic binary states
				[hidSamples_n, hidActivs_n] = RBM.SampleHidGivenVis_S(weights, hidBiases, ...
					hidLinear, visActivs_n);
				visActivs_n = RBM.PropVisFromHid_S(weights, visBiases, visLinear, ...
					hidSamples_n);
				
				%hidActivs_n = obj.PropHidFromVis(visActivs_n);
				%visActivs_n = obj.PropVisFromHid(hidActivs_n);
				
				if i == 1
					hidActivs_0 = hidActivs_n;
				end
			end
			
			% set kth sample to last one in chain
			visActivs_k = visActivs_n;
			
		end
		
		% [hidSample,hidActivs] = SampleHidGivenVis_S(weights, hidBiases, ...
		%		hidLinear, visActivs)
		% - static version of SampleHidGivenVis with 'weights' (DxF) between the 
		% layers, 'hidBiases' (1xF), and logical 'hidLinear' scalar
		function [hidSample,hidActivs] = SampleHidGivenVis_S(weights, hidBiases, ...
				hidLinear, visActivs)
			
			hidActivs = RBM.PropHidFromVis_S(weights, hidBiases, hidLinear, visActivs);
			
			% if the units are [0 1]-valued units
			if ~hidLinear
				hidSample = double(rand(size(hidActivs)) < hidActivs);
			
			else % if the units are linear units, assume unit variance
				hidSample = hidActivs + randn(size(hidActivs));
			end
		end
		
		% [visSample,visActivs] = SampleVisGivenVis_S(weights, hidBiases, ...
		%		hidLinear, visActivs)
		% - static version of SampleHidGivenVis with 'weights' (DxF) between the 
		% layers, 'visBiases' (1xD), and logical 'visLinear' scalar
		function [visSample,visActivs] = SampleVisGivenHid_S(weights, visBiases, ...
				visLinear, hidActivs)
			
			visActivs = RBM.PropVisFromHid_S(weights, visBiases, visLinear, hidActivs);
			
			% if the units are [0 1]-valued units
			if ~visLinear
				visSample = double(rand(size(visActivs)) < visActivs);
				
			else  % if the units are linear units, assume unit variance
				visSample = visActivs + randn(size(hidActivs));
			end
			
		end
		
		% hidActivs = PropHidFromVis_S(weights, hidBiases, hidLinear, visActivs)
		% - static version of PropHidGivenVis with 'weights' (DxF) between the 
		% layers, 'hidBiases' (1xF), and logical 'hidLinear' scalar
		function hidActivs = PropHidFromVis_S(weights, hidBiases, hidLinear, visActivs)
			
			hidActivs = visActivs * weights + repmat(hidBiases,size(visActivs,1),1);
						
			% if the units are [0 1]-valued units
			if ~hidLinear
				hidActivs = NNLayer.Sigm(hidActivs);
			end
			
		end
		
		% visActivs = PropVisFromHid_S(weights, visBiases, visLinear, hidActivs)
		% - static version of PropHidGivenVis with 'weights' (DxF) between the 
		% layers, 'visBiases' (1xD), and logical 'visLinear' scalar
		function visActivs = PropVisFromHid_S(weights, visBiases, visLinear, hidActivs)
			
			visActivs = hidActivs * weights' + repmat(visBiases,size(hidActivs,1),1);
				
			if ~visLinear % if the units are [0 1]-valued units
				visActivs = NNLayer.Sigm(visActivs);
			end
		end
		
	end
	
	methods 
		
		% the Get method for the dependent property 'params'
		function params = get.params(obj)
			
			params = struct;
			params.numEpochs = obj.numEpochs;
			params.learningRate = obj.learningRate;
			params.miniBatchSize = obj.miniBatchSize;
			params.numCDIters = obj.numCDIters;
			params.momentum = obj.momentum;
			params.weightCost = obj.weightCost;
			params.initDist = obj.initDist;
			params.initSpread = obj.initSpread;
			params.verbosity = obj.verbosity;
			params.outDir = obj.outDir;
			params.statusFile = obj.statusFile;
			params.modelFile = obj.modelFile;
			
		end
		
	end
    
	% PROTECTED METHODS ---------------------------------------------------
	methods (Access = protected)
		
		%	SetParams(params)
		%	- sets the parameters of the model (as done in the constructor)
		function SetParams(obj,params) 
			
			% get the parameters set in 'params'
			fields = fieldnames(params);
			
			% set default initSpread
			if strcmp(obj.initDist,'uniform')
				obj.initSpread = 4*sqrt(6/(obj.numHid + obj.numVis));
			elseif strcmp(obj.initDist,'normal')
				obj.initSpread = 0.1;
			end	
			
			% for each parameter field
			for fi = 1:length(fields)
				
				% build command to set param
				cmd = sprintf('obj.%s = params.%s;',fields{fi},fields{fi});
				
				% execute command
				try
					eval(cmd);
				catch e
					obj.MsgOut(sprintf(['Error setting param %s : %s, using ' ...
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
			
			% make sure learning rate, momentum, and weightCost are all of
			% at least length numEpochs
			if length(obj.learningRate) < obj.numEpochs
				obj.MsgOut(sprintf(['Fatal Error: learning rate vector (length = ' ...
					'%d) must be at least of length %d (numEpochs)\n'], ...
					length(obj.learningRate), obj.numEpochs),1);
			end
			
			if length(obj.momentum) < obj.numEpochs
				obj.MsgOut(sprintf(['Fatal Error: momentum vector (length = ' ...
					'%d) must be at least of length %d (numEpochs)\n'], ...
					length(obj.momentum), obj.numEpochs),1);
			end
			
			if length(obj.weightCost) < obj.numEpochs
				obj.MsgOut(sprintf(['Fatal Error: weight cost vector (length = ' ...
					'%d) must be at least of length %d (numEpochs)\n'], ...
					length(obj.weightCost), obj.numEpochs),1);
			end
			
			
			obj.SetInternalParams();
			
		end
		
		%	SetInternalParams()
		%	- sets the filename (statusFile and modelFile) based on the obj.outDir
		%	parameter
		function SetInternalParams(obj)
			
			if isempty(obj.strID)
				obj.strID = 'rbm';
			else
				obj.strID = [obj.strID '.rbm'];
			end
			
			% set the rbm file path
			obj.modelFile = [obj.outDir obj.strID '_obj.mat'];
			
			% set the output status file path, if necessary
			if obj.verbosity == 2
				obj.statusFile = [obj.outDir obj.strID '_status.txt']; 
			end
			
		end
		
		% Update(visSamples_0, e)
		% - updates the weights and biases of the model for inputs visSamples
		function Update(obj, visSamples_0, e)
			
			% do contrastive divergence for k iterations
			[visActivs_k, hidActivs_0] = obj.GibbsSample(visSamples_0, ...
				obj.numCDIters);
			
			% for now, treat the activations as the samples
			visSamples_k = visActivs_k;
			
			% sample from the kth hidden states
			%[hidSamples_k, hidActivs_k] = obj.SampleHidGivenVis(visSamples_k);
			hidActivs_k = obj.PropHidFromVis(visSamples_k);
			
			% calculate deltas
			obj.weightDeltas = (((visSamples_0' * hidActivs_0 - visSamples_k' * ...
				hidActivs_k) / obj.miniBatchSize) - obj.weightCost(e) * ...
				obj.weights) * obj.learningRate(e) + obj.momentum(e) * ...
				obj.weightDeltas;
			
			obj.hidBiasDeltas = (mean(hidActivs_0) - mean(hidActivs_k)) * ...
				obj.learningRate(e) - obj.weightCost(e) * obj.hidBiases + ...
				obj.momentum(e) * obj.hidBiasDeltas;
			
			obj.visBiasDeltas = (mean(visSamples_0) - mean(visSamples_k)) * ...
				obj.learningRate(e) - obj.weightCost(e) * obj.visBiases + ...
				+ obj.momentum(e) * obj.visBiasDeltas;
			
			% update params
			obj.weights = obj.weights + obj.weightDeltas;
			obj.hidBiases = obj.hidBiases + obj.hidBiasDeltas;
			obj.visBiases = obj.visBiases + obj.visBiasDeltas;
			
			%rmsError = obj.GetRMSE(visSamples_0,visSamples_k)
			
			
		end
		
		%	AtEpoch(varargin)
		%	- AtEpoch is called after each epoch and contains any code to be executed
		%	after each epoch (debugging or otherwise)
		function AtEpoch(obj,trainSamples) %#ok<INUSD>
			
		end
		
  end
    
	% PRIVATE METHODS -----------------------------------------------------
	methods (Access = private)
	end
    
    
end