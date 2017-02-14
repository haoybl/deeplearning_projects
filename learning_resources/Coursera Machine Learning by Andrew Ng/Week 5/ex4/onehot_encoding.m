function [ y_out ] = onehot_encoding( y_in, num_labels )
%ONEHOT_ENCODING Summary of this function goes here
%   Detailed explanation goes here

% performs one-hot encoding of class y

y_out = 1 : 1 : num_labels;

y_out = repmat(y_out, [size(y_in, 1), 1]);

y_in = repmat(y_in, [1, num_labels]);

y_out = y_out == y_in;


end

