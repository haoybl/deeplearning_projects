% This scirpt is mainly to expand the training data from orignal images.

clear;
clc;

%% load original image data
load('original_image_data.mat');

%% resize image
for ii = 1:size(giardia_img_data, 1)
   giardia_img_data_resize(ii, :, :) = imresize(squeeze(giardia_img_data(ii, :, :)), 0.25);  %#ok<SAGROW>
end

for ii = 1:size(defects_img_data, 1)
   defects_img_data_resize(ii, :, :) = imresize(squeeze(defects_img_data(ii, :, :)), 0.25);  %#ok<SAGROW>
end

for ii = 1:size(crypto_img_data, 1)
   crypto_img_data_resize(ii, :, :) = imresize(squeeze(crypto_img_data(ii, :, :)), 0.25);  %#ok<SAGROW>
end

giardia_img_data = giardia_img_data_resize;
defects_img_data = defects_img_data_resize;
crypto_img_data = crypto_img_data_resize;
clear('giardia_img_data_resize', 'defects_img_data_resize', 'crypto_img_data_resize');

%% expand by rotation and flip+rotation
% ================= giardia data ====================================
giardia_rotate = giardia_img_data;
for ii = 1:size(giardia_img_data, 1)
   ii
   img_data = squeeze(giardia_img_data(ii, :, :));
   expanded_img_data = expand_by_rotation(img_data);
   giardia_rotate = cat(1, giardia_rotate, expanded_img_data); 
end

giardia_flip = [];
for ii = 1:size(giardia_img_data, 1)
   ii
   img_data = squeeze(giardia_img_data(ii, :, :));
   giardia_flip(ii, :, :) = flipud(img_data); %#ok<SAGROW>
end

giardia_flip_rotate = giardia_flip;
for ii = 1:size(giardia_flip, 1)
   ii
   img_data = squeeze(giardia_flip(ii, :, :));
   expanded_img_data = expand_by_rotation(img_data);
   giardia_flip_rotate = cat(1, giardia_flip_rotate, expanded_img_data);
end

clear('ii', 'img_data', 'expanded_img_data', 'giardia_flip');
save('giardia_expanded.mat', 'giardia_flip_rotate', 'giardia_rotate', '-v7.3');

% ================= defects data =======================================
defects_rotate = defects_img_data;
for ii = 1:size(defects_img_data, 1)
   ii
   img_data = squeeze(defects_img_data(ii, :, :));
   expanded_img_data = expand_by_rotation(img_data);
   defects_rotate = cat(1, defects_rotate, expanded_img_data); 
end

defects_flip = [];
for ii = 1:size(defects_img_data, 1)
   ii
   img_data = squeeze(defects_img_data(ii, :, :));
   defects_flip(ii, :, :) = flipud(img_data); %#ok<SAGROW>
end

defects_flip_rotate = defects_flip;
for ii = 1:size(defects_flip, 1)
   ii
   img_data = squeeze(defects_flip(ii, :, :));
   expanded_img_data = expand_by_rotation(img_data);
   defects_flip_rotate = cat(1, defects_flip_rotate, expanded_img_data);
end

clear('ii', 'img_data', 'expanded_img_data', 'defects_flip');
save('defects_expanded.mat', 'defects_flip_rotate', 'defects_rotate', '-v7.3');

% ================= crypto data =======================================
crypto_rotate = crypto_img_data;
for ii = 1:size(crypto_img_data, 1)
   ii
   img_data = squeeze(crypto_img_data(ii, :, :));
   expanded_img_data = expand_by_rotation(img_data);
   crypto_rotate = cat(1, crypto_rotate, expanded_img_data); 
end

crypto_flip = [];
for ii = 1:size(crypto_img_data, 1)
   ii
   img_data = squeeze(crypto_img_data(ii, :, :));
   crypto_flip(ii, :, :) = flipud(img_data); %#ok<SAGROW>
end

crypto_flip_rotate = crypto_flip;
for ii = 1:size(crypto_flip, 1)
   ii
   img_data = squeeze(crypto_flip(ii, :, :));
   expanded_img_data = expand_by_rotation(img_data);
   crypto_flip_rotate = cat(1, crypto_flip_rotate, expanded_img_data);
end

clear('ii', 'img_data', 'expanded_img_data', 'crypto_flip');
save('crypto_expanded.mat', 'crypto_flip_rotate', 'crypto_rotate', '-v7.3');