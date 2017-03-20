%% Extract Features from the Images
clear;
close all

%% load data
load('original_img.mat')

NoImgG = size(Giardia_Image, 1);    %number of giardia images
NoImgC = size(Crypto_Image, 1);        %number of cryptosporidium images
NoImgD = size(Defects_Image, 1);       %number of defect images

FEATURE = zeros(6,NoImgC+NoImgG+NoImgD);
% FEATURE(1,:) for class (1 for giardia, 2 for ctypto, 3 for defects)
% FEATURE(2,:) for MMR
% FEATURE(3,:) for SO
% FEATURE(4,:) for CO
FEATURE(1:3,1:NoImgG)=repmat([1;0;0], [1, NoImgG]);
FEATURE(1:3,NoImgG+1:NoImgG+NoImgC)=repmat([0;1;0], [1, NoImgC]);
FEATURE(1:3,NoImgG+NoImgC+1:end)=repmat([0;0;1], [1, NoImgD]);

%% extract features for each image
for ii = 1:(NoImgC+NoImgG+NoImgD)
    if ii > NoImgG+NoImgC
        kk = ii - NoImgG - NoImgC;
        IMG_in = uint8( squeeze(Defects_Image(kk,:,:)) );
    else if ii > NoImgG
        kk = ii - NoImgG;
        IMG_in = uint8( squeeze(Crypto_Image(kk,:,:)) );
        else
            kk = ii;
            IMG_in = uint8( squeeze(Giardia_Image(kk,:,:)) );
        end
    end
    
% IMG_in = uint8( squeeze(Giardia_Image(30,:,:)) );
% IMG_in = imread('IMG0091.bmp');
% figure,imshow(IMG_in);
% title('Original Image');
ImageSize = size(IMG_in);%[488,648]

%% segmentation 
% % segmentation using active contour: be careful about the mask !
% % for one image: ~7 sec
% mask = false(ImageSize);
% dx = uint8(ImageSize(1)/4);%488/4
% dy = uint8(ImageSize(2)/4);%648/4
% mask(dx+1:end-dx,dy+1:end-dy) = true;
% NoIt=500; %number of iterations
% % IMG_seg = activecontour(IMG_in,mask,NoIt,'edge',0.02);
% IMG_seg = activecontour(IMG_in,mask,NoIt);
% figure, imshow(IMG_seg);
% title('Segmented Image by Active Contour');
% % active contour method:'Chan-Vese'(default),'edge'
% % BW = activecontour(A, MASK, N, METHOD, SMOOTHFACTOR)
% % default SMOOTHFACTOR = 1 for 'edge' and 0 for 'Chan-Vese'. 
% segmentation with Otsu's threshold
% for one image: ~1 sec
Level = graythresh(IMG_in);
IMG_seg = im2bw(IMG_in,Level);
% figure, imshow(IMG_seg);
% title('Segmented Image by Otsu Thresholding

%% region properties
MajorAxisLengths = regionprops(IMG_seg,'MajorAxisLength');
MinorAxisLengths = regionprops(IMG_seg,'MinorAxisLength');
Perimeters = regionprops(IMG_seg,'Perimeter');
Areas = regionprops(IMG_seg,'Area');
Solidities = regionprops(IMG_seg,'Solidity');
Centroids = regionprops(IMG_seg,'Centroid');
FilledImages = regionprops(IMG_seg,'FIlledImage');

%% calculating features
NoObject = length(Areas);
MaxArea = Areas(1,1).Area;
for jj = 1:NoObject
    if Areas(jj,1).Area >= MaxArea
        MaxArea = Areas(jj,1).Area;
        IndexBiggestObject = jj;
    end
end
A_obj = MaxArea;
MajorAL_obj = MajorAxisLengths(IndexBiggestObject).MajorAxisLength;
MinorAL_obj = MinorAxisLengths(IndexBiggestObject).MinorAxisLength;
MMratio_obj = MajorAL_obj/MinorAL_obj;%FEATURE(2,:)
FEATURE(4,ii) = MMratio_obj;
So_obj = Solidities(IndexBiggestObject).Solidity;%FEATURE(3,:)
FEATURE(5,ii) = So_obj;
Pr_obj = Perimeters(IndexBiggestObject).Perimeter;
Co_obj = 4*pi*A_obj/(Pr_obj^2);%FEATURE(4,:) compactness=1/circularity
FEATURE(6,ii) = Co_obj;
Ct_obj = Centroids(IndexBiggestObject).Centroid;

%%
% %% extract and fill the larggest object/component
% Filled_obj = FilledImages(IndexBiggestObject).FilledImage;
% figure, imshow(Filled_obj);
% title('Filled Image of the biggest object in the Segmenting Result');
% % SE = strel('square',3); %structuring element

% %% outlining
% IMG_outline = bwmorph(IMG_seg,'remove');
% figure, imshow(IMG_outline);
% title('Image Outline');

% %% skeletonization
% IMG_skel = bwmorph(IMG_seg,'skel',Inf);
% figure, imshow(IMG_skel);
% title('Image Skeleton');

% %% removal of isolated pixels
% IMG_remiso = bwmorph(IMG_outline,'clean');

% %% suppress light structures connected to image border
% IMG_clrbor = imclearborder(IMG_remiso,8);
% figure, imshow(IMG_clrbor);

% hold on
% plot(Ct_obj(1,1),Ct_obj(1,2),'r');
% hold off

end

TARGET = FEATURE(1:3,:);
INPUT = FEATURE(4:6,:);
save('img_feature.mat','FEATURE');
save('img_input.mat','INPUT');
save('img_target.mat','TARGET');
