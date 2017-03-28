% Prepare images data used for deep learning from original giardia, 
% Defects and Crypto Data
clear;
clc;

%% Load image Data

% load Giardia images
<<<<<<< HEAD
imagefiles = dir('Giardia/*.bmp');
=======
imagefiles = dir('Giardia/*.bmp');      
>>>>>>> refs/remotes/origin/master
nfiles = length(imagefiles);    % Number of files found
for ii=1:nfiles
   currentfilename = imagefiles(ii).name;
   currentimage = imread(currentfilename);
   giardia_img_data(ii, :, :) = double(currentimage(45:444, 125:524)); %#ok<SAGROW>
end

% load Defects images
imagefiles = dir('Defects/*.bmp');      
nfiles = length(imagefiles);    % Number of files found
for ii=1:nfiles
   currentfilename = imagefiles(ii).name;
   currentimage = imread(currentfilename);
   defects_img_data(ii, :, :) = double(currentimage(45:444, 125:524)); %#ok<SAGROW>
end

% load Crypto images
imagefiles = dir('Crypto/*.bmp');      
nfiles = length(imagefiles);    % Number of files found
for ii=1:nfiles
   currentfilename = imagefiles(ii).name;
   currentimage = imread(currentfilename);
   crypto_img_data(ii, :, :) = double(currentimage(45:444, 125:524)); %#ok<SAGROW>
end

clear('currentfilename', 'currentimage', 'ii', 'imagefiles', 'nfiles');





