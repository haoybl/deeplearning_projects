for i = 1: 20
    
count = 1;
figure;
for row = 1:4
    for col = 1:6
                
        subplot(4, 6, count);
        index = randi([1000000, 2000000], 1);
        histogram(train_data_reshaped(index, :), 128)
        count = count + 1;       
        
    end
end

end