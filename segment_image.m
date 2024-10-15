function [seg] = segment_image2(I)
% SEGMENT_IMAGE2 Segments the input color image into boundaries focusing on main objects.
% This function adjusts its parameters and processing steps to capture high-level objects
% and minimize unnecessary details and background.

    % Ensure the image is in double precision
    if ~isa(I, 'double')
        I = im2double(I);
    end

    % Convert the image to LAB color space
    lab_image = rgb2lab(I);

    % Estimate the image type
    image_type = estimate_image_type(I);

    % Set segmentation parameters based on image type
    switch image_type
        case 'animals'
            sigma = 5;                % High smoothing to merge textures
            nColors = 2;              % Two clusters to separate object and background
            cannyThresh = [0.05 0.15];% Lower thresholds to capture object boundaries
        case 'people'
            sigma = 5;                % High smoothing to merge clothing colors
            nColors = 2;              % Two clusters to separate person and background
            cannyThresh = [0.05 0.15];% Lower thresholds to capture object boundaries
        case 'buildings'
            sigma = 3;                % Moderate smoothing
            nColors = 2;              % Two clusters to separate building and background
            cannyThresh = [0.1 0.3];  % Moderate thresholds
        case 'nature'
            sigma = 6;                % High smoothing to merge natural elements
            nColors = 2;              % Two clusters for foreground and background
            cannyThresh = [0.1 0.3];  % Moderate thresholds
        otherwise  % Default parameters
            sigma = 4;
            nColors = 2;
            cannyThresh = [0.1 0.3];
    end

    % Apply Gaussian smoothing
    lab_smooth = imgaussfilt(lab_image, sigma);

    % Perform color-based segmentation using k-means
    ab = lab_smooth(:,:,2:3);  % Use only a and b channels (color)
    ab = reshape(ab, [], 2);
    [cluster_idx, ~] = kmeans(ab, nColors, 'Distance', 'sqEuclidean', 'Replicates', 3);

    % Reshape the cluster map to the original image size
    pixel_labels = reshape(cluster_idx, size(lab_smooth, 1), size(lab_smooth, 2));

    % Identify the cluster corresponding to the foreground object
    % Assume that the cluster with higher variance in the L channel is the object
    L_channel = lab_smooth(:,:,1);
    object_cluster = identify_object_cluster(pixel_labels, L_channel);

    % Create a binary mask for the object
    object_mask = pixel_labels == object_cluster;

    % Perform edge detection on the object mask
    edges_object = edge(object_mask, 'Canny');

    % Perform edge detection on the L channel
    edges_canny = edge(L_channel, 'Canny', cannyThresh);

    % Combine edge maps
    edges_combined = edges_object | edges_canny;

    % Apply morphological operations to clean up the edges
    edges_cleaned = imfill(edges_combined, 'holes');  % Fill enclosed regions
    edges_cleaned = bwareaopen(edges_cleaned, 50);    % Remove small objects

    % Return the cleaned edge map as seg
    seg = edges_cleaned;
end

function image_type = estimate_image_type(I)
% ESTIMATE_IMAGE_TYPE Estimates the image type based on simple image features.
% Returns one of 'animals', 'people', 'buildings', 'nature'.

    % Simple heuristic based on aspect ratio and color variance
    [rows, cols, ~] = size(I);
    aspect_ratio = cols / rows;
    color_variance = mean(var(reshape(I, [], 3)));

    if color_variance < 0.02
        image_type = 'buildings';
    elseif aspect_ratio > 1.5
        image_type = 'nature';
    else
        % Assuming 'people' or 'animals' based on the presence of skin tones
        % Convert to HSV and check for skin-like colors
        hsvI = rgb2hsv(I);
        hue = hsvI(:,:,1);
        saturation = hsvI(:,:,2);
        value = hsvI(:,:,3);

        skin_mask = (hue > 0.0 & hue < 0.1) & (saturation > 0.2 & saturation < 0.7) & (value > 0.4 & value < 0.9);
        skin_ratio = sum(skin_mask(:)) / numel(skin_mask);

        if skin_ratio > 0.1
            image_type = 'people';
        else
            image_type = 'animals';
        end
    end
end

function object_cluster = identify_object_cluster(pixel_labels, L_channel)
% IDENTIFY_OBJECT_CLUSTER Identifies which cluster corresponds to the foreground object.
% Assumes that the object cluster has higher variance in the L channel.

    clusters = unique(pixel_labels);
    num_clusters = length(clusters);
    cluster_variances = zeros(num_clusters,1);

    for i = 1:num_clusters
        cluster_mask = pixel_labels == clusters(i);
        cluster_variances(i) = var(L_channel(cluster_mask));
    end

    [~, idx] = max(cluster_variances);
    object_cluster = clusters(idx);
end
