function [ output ] = one_hot_with_total_num( label, num_class )
%ONE_HOT_WITH_TOTAL_NUM Summary of this function goes here
%   Detailed explanation goes here

    output = zeros(1, num_class);
    
    output(label) = 1;


end

