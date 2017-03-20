function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

%% Prepare param combinations for grid search

C = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
sigma = C;

[C, sigma] = ndgrid(C, sigma);
param_comb = [C(:), sigma(:)];
cost = zeros(size(param_comb, 1), 1);

for pi = 1:size(param_comb, 1)
    
    C = param_comb(pi, 1);
    sigma = param_comb(pi, 2);
    model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
    predictions = svmPredict(model, Xval);
    cost(pi) = mean(double(predictions ~= yval));
end

%% find optimal C and sigma

[~, min_i] = min(cost);

C = param_comb(min_i, 1);
sigma = param_comb(min_i, 2);

% =========================================================================

end
