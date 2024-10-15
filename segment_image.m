function [seg] = segment_image(I)
%SEGMENT_IMAGE Segments the input color image into regions using K-means clustering.
%
% Our method segments the input color image using K-means clustering applied to a
% feature space that combines color and spatial information. Each pixel is represented
% by its RGB color values and its normalized spatial coordinates (x, y). The spatial
% coordinates are weighted to control their influence on the clustering. The number of
% clusters K is set to 5. K-means clustering partitions the pixels into K clusters based
% on their feature vectors, resulting in a label map where each pixel is assigned a cluster
% label. This label map constitutes the segmentation of the image.
%
% This code uses only built-in MATLAB functions and methods covered in the module up to
% week 5, specifically K-means clustering and basic image processing techniques.
%
% Author: Maruf Bepary
% Date: 14-10-2024
%

    % Convert image to double precision if necessary
    if ~isa(I, 'double')
        I = im2double(I);
    end

    [rows, cols, channels] = size(I);
    
    % Reshape the image into an N x channels matrix
    pixel_data = reshape(I, rows * cols, channels);
    
    % Create a grid of coordinates
    [X, Y] = meshgrid(1:cols, 1:rows);
    % Reshape the coordinates into N x 1 vectors
    X = X(:);
    Y = Y(:);
    
    % Normalize the color data and spatial data
    pixel_data = pixel_data / max(pixel_data(:));
    X_norm = X / max(X);
    Y_norm = Y / max(Y);
    
    % Weight for spatial coordinates
    spatial_weight = 0.5; % Adjust this parameter as needed
    
    % Combine color and spatial data
    feature_data = [pixel_data, spatial_weight * [X_norm, Y_norm]];
    
    % Number of clusters
    K = 5; % Adjust K as needed
    
    % Run K-means clustering
    % Repeat kmeans to avoid local minima
    [cluster_idx, ~] = kmeans(feature_data, K, 'MaxIter', 200, 'Replicates', 3);
    
    % Reshape cluster_idx into the image
    seg = reshape(cluster_idx, rows, cols);
end

