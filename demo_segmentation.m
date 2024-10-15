clear;           % Clear workspace
clc;             % Clear command window
close all;       % Close any open figure windows


% Read the input image
I = im2double(imread('/MATLAB Drive/images/im1.jpg')); % Replace 'im1.jpg' with your image file

% Call the segmentation function
seg = segment_image(I);

% Display the original image
figure;
subplot(1,2,1);
imshow(I);
title('Original Image');

% Display the segmentation result
subplot(1,2,2);

% Check if seg is a label map or boundary map
unique_values = unique(seg);
if islogical(seg) || (numel(unique_values) == 2 && all(ismember(unique_values, [0,1])))
    % Assume seg is a binary boundary map
    imshow(seg);
    title('Segmentation Boundaries');
else
    % Assume seg is a label map
    % Map labels to colors for visualization
    seg_rgb = label2rgb(seg);
    imshow(seg_rgb);
    title('Segmented Image');
end

