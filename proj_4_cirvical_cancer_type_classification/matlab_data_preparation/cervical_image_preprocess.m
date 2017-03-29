%% CERVICAL_IMAGE_PREPROCESS
% This script is used to load original image, resize, shuffle and save them
% into separate mat files for range finder or classifier. 
%

clear;
clc;

%% Define environmental variables

%resize = [3200 2400, 3];
resize = [3200, 2400];
images_per_mat = 100;

top_path = 'D:\OneDrive\NTU\PhD\PHD DATA\Dataset\Kaggles_Cervical_cancer_screening\train\';

type_1_dir = [top_path, 'Type_1']; 
type_2_dir = [top_path, 'Type_2'];
type_3_dir = [top_path, 'Type_3'];

%% Type 1 
% Load image data
imagefiles = dir([type_1_dir, '\*.jpg']);      
nfiles = length(imagefiles)    % Number of files found
for ii=1:nfiles
   ii
   currentimage = imread([type_1_dir, '\', imagefiles(ii).name]);
   
   % Rotate image
   img_size = size(currentimage);
   if img_size(1)<img_size(2)
       currentimage = rot90(currentimage);
   end
   
   % Resize image
   currentimage = imresize(currentimage, [3200, 2400]);
   type_1_images{ii} = currentimage;
end

no_parts = ceil(nfiles/images_per_mat);

for part_i = 1:no_parts
    
    savename = ['type_1_image_data_part_', num2str(part_i), '.mat'];
    
    if part_i < no_parts
        images_temp = type_1_images(1, (part_i-1)*images_per_mat+1: part_i*images_per_mat);
    else
        images_temp = type_1_images(1, (part_i-1)*images_per_mat+1:end);
    end
    
    save(savename, 'images_temp', '-v7.3');
    
end

clear('type_1_images', 'type_1_dir', 'images_temp');

%% Type 2 
% Load image data
imagefiles = dir([type_2_dir, '\*.jpg']);      
nfiles = length(imagefiles)    % Number of files found
for ii=1:nfiles
   ii
   currentimage = imread([type_2_dir, '\', imagefiles(ii).name]);
   
   % Rotate image
   img_size = size(currentimage);
   if img_size(1)<img_size(2)
       currentimage = rot90(currentimage);
   end
   
   % Resize image
   currentimage = imresize(currentimage, resize);
   type_2_images{ii} = currentimage;
end

no_parts = ceil(nfiles/images_per_mat);

for part_i = 1:no_parts
    
    savename = ['type_2_image_data_part_', num2str(part_i), '.mat'];
    
    if part_i < no_parts
        images_temp = type_2_images(1, (part_i-1)*images_per_mat+1: part_i*images_per_mat);
    else
        images_temp = type_2_images(1, (part_i-1)*images_per_mat+1:end);
    end
    
    save(savename, 'images_temp', '-v7.3');
    
end

clear('type_2_images', 'type_2_dir', 'images_temp');

%% Type 3 
% Load image data
imagefiles = dir([type_3_dir, '\*.jpg']);      
nfiles = length(imagefiles)    % Number of files found
for ii=1:nfiles
   ii
   currentimage = imread([type_3_dir, '\', imagefiles(ii).name]);
   
   % Rotate image
   img_size = size(currentimage);
   if img_size(1)<img_size(2)
       currentimage = rot90(currentimage);
   end
   
   % Resize image
   currentimage = imresize(currentimage, resize);
   type_3_images{ii} = currentimage;
end

no_parts = ceil(nfiles/images_per_mat);

for part_i = 1:no_parts
    
    savename = ['type_3_image_data_part_', num2str(part_i), '.mat'];
    
    if part_i < no_parts
        images_temp = type_3_images(1, (part_i-1)*images_per_mat+1: part_i*images_per_mat);
    else
        images_temp = type_3_images(1, (part_i-1)*images_per_mat+1:end);
    end
    
    save(savename, 'images_temp', '-v7.3');
    
end

clear('type_3_images', 'type_3_dir', 'images_temp');


