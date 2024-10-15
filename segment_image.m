function [seg] = segment_image2(I)
% SEGMENT_IMAGE Segments the input color image into boundaries using a balanced combination of methods.
% This function uses a single set of parameters for all image types.

    % Ensure the image is in double precision
    if ~isa(I, 'double')
        I = im2double(I);
    end

    % Convert the image to LAB color space
    lab_image = rgb2lab(I);

    % Set segmentation parameters
    sigma = 4;  % Moderate smoothing to reduce noise while keeping detail
    nColors = 2;  % Fewer color clusters to group similar regions
    cannyThresh = [0.1 0.3];  % Higher Canny thresholds to focus on sharp edges

    % Apply Gaussian smoothing
    lab_smooth = imgaussfilt(lab_image, sigma);

    % Perform color-based segmentation using k-means
    ab = lab_smooth(:,:,2:3);  % Use only a and b channels (color)
    ab = reshape(ab, [], 2);
    [cluster_idx, ~] = kmeans(ab, nColors, 'Distance', 'sqEuclidean', 'Replicates', 3);

    % Reshape the cluster map to the original image size
    pixel_labels = reshape(cluster_idx, size(lab_smooth, 1), size(lab_smooth, 2));

    % Create a binary edge map from the clustered image
    edges_kmeans = edge(pixel_labels, 'Canny');

    % Perform edge detection on the L channel (luminance)
    L_channel = lab_smooth(:,:,1);
    edges_canny = edge(L_channel, 'Canny', cannyThresh);

    % Combine edge maps
    edges_combined = edges_kmeans | edges_canny;

    % Apply morphological operations to clean up the edges
    se1 = strel('disk', 1);
    edges_cleaned = bwmorph(edges_combined, 'thin', Inf);
    edges_cleaned = imdilate(edges_cleaned, se1);
    edges_cleaned = bwmorph(edges_cleaned, 'spur');
    edges_cleaned = bwareaopen(edges_cleaned, 10);  % Remove small objects

    % Return the cleaned edge map as seg
    seg = edges_cleaned;
end
