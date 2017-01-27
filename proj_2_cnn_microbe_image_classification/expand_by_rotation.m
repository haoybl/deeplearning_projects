function [ expanded_img ] = expand_by_rotation( input_img )
%EXPAND_BY_ROTATION rotate of image counterclockwise by certain angle.
%   Given an image, rotate counterclockwise by certain angle at multiple
%   times until goes back to 360. 

base_angle = 10;
angles = base_angle * (1:35);

for ii=1:35
    expanded_img(ii, :, :) = imrotate(input_img, angles(ii), 'crop', 'bicubic'); %#ok<AGROW>
end

