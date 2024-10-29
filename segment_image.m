function [seg] = segment_image(I)
% Image segmentation using cRF and ncRF mechanisms
% Input: I - RGB image in double format (range [0,1])
% Output: seg - Binary boundary map (0s and 1s)

% Convert RGB to grayscale
if size(I,3) == 3
    Igray = rgb2gray(I);
else
    Igray = I;
end

% Parameters
sigma_small = 3;        % Small scale sigma for complex cells
sigma_large = 5;        % Large scale sigma for hypercomplex cells
lambda = 0.1;
gamma = 0.75;
orientations = [0 45 90 135];

% Initialize response matrices
complexResponses_small = zeros([size(Igray), length(orientations)]);
simpleResponses = zeros([size(Igray), length(orientations)]);
hypercomplexResponses = zeros([size(Igray), length(orientations)]);
contextualEnhancement = zeros([size(Igray), length(orientations)]);
contextualSuppression = zeros([size(Igray), length(orientations)]);

% Process each orientation
for i = 1:length(orientations)
    theta = orientations(i);
    
    % Simple cells (phase = 90)
    gaborFilterSimple = gabor2(sigma_small, lambda, theta, gamma, 90);
    simpleResponse = conv2(Igray, gaborFilterSimple, 'same');
    simpleResponses(:,:,i) = abs(simpleResponse);  % Edge strength
    
    % Complex cells at small scale (sigma_small)
    gaborFilter90_small = gabor2(sigma_small, lambda, theta, gamma, 90);
    gaborFilter0_small = gabor2(sigma_small, lambda, theta, gamma, 0);
    response90_small = conv2(Igray, gaborFilter90_small, 'same');
    response0_small = conv2(Igray, gaborFilter0_small, 'same');
    complexResponse_small = sqrt(response90_small.^2 + response0_small.^2);
    complexResponses_small(:,:,i) = complexResponse_small;
    
    % Complex cells at large scale (sigma_large) for hypercomplex cells
    gaborFilter90_large = gabor2(sigma_large, lambda, theta, gamma, 90);
    gaborFilter0_large = gabor2(sigma_large, lambda, theta, gamma, 0);
    response90_large = conv2(Igray, gaborFilter90_large, 'same');
    response0_large = conv2(Igray, gaborFilter0_large, 'same');
    complexResponse_large = sqrt(response90_large.^2 + response0_large.^2);
    
    % Hypercomplex response (end-stopping)
    hypercomplexResponse = complexResponse_small - complexResponse_large;
    hypercomplexResponse(hypercomplexResponse < 0) = 0;  % Threshold negative values
    hypercomplexResponses(:,:,i) = hypercomplexResponse;
    
    % Contextual Enhancement (Collinear Facilitation)
    % Create an elongated Gaussian kernel aligned along the orientation
    kernelSize = 15;
    enhancementKernel = orientedGaussianKernel(kernelSize, theta, 10, 1);
    % Convolve the complex response with the enhancement kernel
    enhancedResponse = conv2(complexResponse_small, enhancementKernel, 'same');
    contextualEnhancement(:,:,i) = enhancedResponse;
    
    % Contextual Suppression (Surround Suppression)
    % Apply isotropic Gaussian blur to the complex response
    suppressionKernel = fspecial('gaussian', [15 15], 5);
    suppressedResponse = complexResponse_small - imfilter(complexResponse_small, suppressionKernel, 'same');
    suppressedResponse(suppressedResponse < 0) = 0;  % Threshold negative values
    contextualSuppression(:,:,i) = suppressedResponse;
end

% Combine responses across orientations
maxSimpleResponse = max(simpleResponses, [], 3);
maxComplexResponse = max(complexResponses_small, [], 3);
maxHypercomplexResponse = max(hypercomplexResponses, [], 3);
maxContextualEnhancement = max(contextualEnhancement, [], 3);
maxContextualSuppression = max(contextualSuppression, [], 3);

% Combine simple, complex, hypercomplex, and ncRF responses
% Adjust weights as needed (weights sum to 1)
w1 = 0.1;  % Weight for simple cells
w2 = 0.4;  % Weight for complex cells
w3 = 0.2;  % Weight for hypercomplex cells
w4 = 0.15; % Weight for contextual enhancement
w5 = 0.15; % Weight for contextual suppression
combinedResponse = w1 * maxSimpleResponse + w2 * maxComplexResponse + ...
                   w3 * maxHypercomplexResponse + w4 * maxContextualEnhancement + ...
                   w5 * maxContextualSuppression;

% Normalize response to [0,1] range
combinedResponse = (combinedResponse - min(combinedResponse(:))) / ...
                  (max(combinedResponse(:)) - min(combinedResponse(:)));

% Apply non-maximum suppression to thin edges
segedges = edge(combinedResponse, 'canny', [0.1 0.2]);

% Clean up boundaries
seg = bwmorph(segedges, 'thin', Inf);  % Ensure single-pixel width
seg = bwareaopen(seg, 20);             % Remove small segments

% Close small gaps
se = strel('disk', 1);
seg = imclose(seg, se);

end

function gb = gabor2(sigma, freq, orient, aspect, phase)
% Implementation of 2D Gabor filter
% Parameters:
% sigma  = standard deviation of Gaussian envelope
% freq   = frequency of sine wave
% orient = orientation from horizontal (degrees)
% aspect = aspect ratio of Gaussian envelope
% phase  = phase of sine wave (degrees)

sz = fix(7 * sigma / max(0.2, aspect));
if mod(sz, 2) == 0, sz = sz + 1; end

[x, y] = meshgrid(-fix(sz/2):fix(sz/2));

% Rotation 
orient = (orient - 90) * pi / 180;
xDash = x * cos(orient) + y * sin(orient);
yDash = -x * sin(orient) + y * cos(orient);

phase = phase * pi / 180;

gb = exp(-0.5 * ((xDash.^2 / sigma^2) + (aspect^2 * yDash.^2 / sigma^2))) .* ...
     cos(2 * pi * xDash * freq + phase);

% Normalize filter
pos_sum = sum(gb(gb > 0));
neg_sum = sum(-gb(gb < 0));

if pos_sum ~= 0
    gb(gb > 0) = gb(gb > 0) / pos_sum;
end
if neg_sum ~= 0
    gb(gb < 0) = gb(gb < 0) / neg_sum;
end
end

function kernel = orientedGaussianKernel(size, orientation, sigma_x, sigma_y)
% Creates an anisotropic Gaussian kernel oriented along a specific angle
% size: size of the kernel (should be odd)
% orientation: orientation angle in degrees
% sigma_x: standard deviation along x
% sigma_y: standard deviation along y

% Create coordinate grids
halfSize = floor(size/2);
[x, y] = meshgrid(-halfSize:halfSize, -halfSize:halfSize);

% Convert orientation to radians
theta = -orientation * pi / 180;

% Rotate coordinates
x_theta = x * cos(theta) - y * sin(theta);
y_theta = x * sin(theta) + y * cos(theta);

% Compute Gaussian function
kernel = exp(- (x_theta.^2 / (2 * sigma_x^2) + y_theta.^2 / (2 * sigma_y^2)));

% Normalize kernel
kernel = kernel / sum(kernel(:));
end

