% Save 2D Image Pixel Values into 1D vetor ~4.3 sec
clear all
close all

load('original_img.mat');

NoImgG = size(Giardia_Image, 1);    %number of giardia images
NoImgC = size(Crypto_Image, 1);        %number of cryptosporidium images
NoImgD = size(Defects_Image, 1);       %number of defect images

INPUT_IMG = [];
INPUT_TARGET = [];

% Read Giardia_Images
for ii = 1: NoImgG
   temp = uint8( squeeze(Giardia_Image(ii,:,:)));
   temp = imresize(temp,0.25);
   INPUT_IMG(ii, :) = reshape(temp, [1, numel(temp)]); %#ok<*SAGROW>
   INPUT_TARGET(ii, :) = [1 0 0 ];
end

% Read Crypto_Images
for ii = 1:NoImgC
   temp = uint8( squeeze(Crypto_Image(ii,:,:)));
   temp = imresize(temp,0.25);
   INPUT_IMG(NoImgG + ii, :) = reshape(temp, [1, numel(temp)]); %#ok<*SAGROW>
   INPUT_TARGET(NoImgG + ii, :) = [0 1 0 ];
end

% Read Defect Images
for ii = 1:NoImgD
   temp = uint8( squeeze(Defects_Image(ii,:,:)));
   temp = imresize(temp,0.25);
   INPUT_IMG(NoImgG + NoImgC + ii, :) = reshape(temp, [1, numel(temp)]); %#ok<*SAGROW>
   INPUT_TARGET(NoImgG + NoImgC + ii, :) = [0 0 1 ];
end


save('img_input_1D.mat','INPUT_IMG', 'INPUT_TARGET');
%19764x412 uint8 matrix, 412 columns, 19764 rows