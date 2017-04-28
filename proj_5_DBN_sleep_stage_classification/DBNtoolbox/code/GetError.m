function [error, dV] = GetError(weightsVect, weightsDims, hidLinear, ...
				input, varargin)
			
		Sigm = @(x) 1 ./ (1 + exp(-x));
			
		% target labels are also passed in, assume we're working with a
		% classifier
		if length(varargin) >= 1
			targets = varargin{1};
			isClassif = 1;
		else
			isClassif = 0;
		end

		% convert weightsVect into weights and biases cell arrays
		[w, b] = DeepNN.Vect2Weights(weightsVect,weightsDims);

		% propogate the activiations through the network
		layerActivs = cell(1,size(weightsDims,1));
		layerInput = input;
		for i = 1:size(weightsDims,1)
			if hidLinear(i)
				layerActivs{i} = layerInput * w{i} + repmat(b{i}, ...
					size(input,1),1);
			else
				layerActivs{i} = Sigm(layerInput * w{i} + ...
					repmat(b{i},size(input,1),1));
			end
			layerInput = layerActivs{i};
		end

		% softmax output if it's a classifier
		if isClassif
			layerActivs{end} = DeepNN.Softmax(layerActivs{end});
		end

		% calc cross-entropy error
		if ~isClassif
			error = -sum(sum(input .* log(layerActivs{end}) + (1-input) .* ...
				log(1-layerActivs{end}))) / size(layerInput,1);
		else
			error = -sum(sum(targets .* log(layerActivs{end})));
		end

		% calc derivatives and then vectorize 
		if length(varargin) >= 1
			[dW, dB] = CalcDerivs(input, layerActivs, w, b, targets);
		else
				[dW, dB] = CalcDerivs(input, layerActivs, w, b);
		end

		dV = DeepNN.Weights2Vect(dW,dB)';

end
	
function [dW, dB] = CalcDerivs(inputData, layerActivs, weights, biases, ...
				varargin)
			
	if length(varargin) >= 1
		isClassif = 1;
		labels = varargin{1};
	else
		isClassif = 0;
	end

	% init weight and bias derivatives
	dW = cell(size(weights));
	dB = cell(size(biases));

	% initilize the current deltas to be the difference between the
	% reconstructed data and the real data 
	if ~isClassif
		curDeltas = layerActivs{end} - inputData;

		% so that later sum in mat mult is an average
		curDeltas = curDeltas / size(inputData,1); 
	else
		curDeltas = layerActivs{end} - labels;
	end

	% calcualte the derivates of the weights and biases for each layer,
	% moving backwards through the DBN layers
	for i = length(dW):-1:1

		% if we're not in the first layer
		if i > 1

			% calc weight derivs for this layer from the product of the
			% previous layer's activations and this layer's deltas
			dW{i} = layerActivs{i-1}' * curDeltas;

			% calc average delta for each dimensiom for biases
			dB{i} = sum(curDeltas);

			% update the curDeltas for the next earlier layer

			% if in the top/middle layer of a recon DeepNN
			if i == length(dW)/2 + 1 && ~isClassif

				% keep weights bi-directional
				curDeltas = curDeltas * weights{i}';
			else

				% also mulitply by the chained derivative of the next/
				% earlier sigmoidal hidden layer 
				curDeltas = curDeltas * weights{i}' .* layerActivs{i-1} ...
					.* (1-layerActivs{i-1});
			end

		else % in the first layer

			% use input data now as for derivative product
			dW{i} = inputData' * curDeltas;

			% calc average delta for each dimensiom for biases
			dB{i} = sum(curDeltas);
		end
	end
end