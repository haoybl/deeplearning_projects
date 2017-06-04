function class_counter( one_hot_labels, hist_title)
%CLASS_COUNTER 
%   Generate histogram of 5 sleep classes

    % Convert one hot labels back to single values
    [row, col] = find(one_hot_labels);
    labels(row) = col;

    % Genereate histogram
    figure;
    histogram(labels);
    title(hist_title);

end

