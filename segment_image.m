function [seg] = segment_image(I)
% Image segmentation using complex cells (Gabor filters) based on lab implementation
% Input: I - RGB image in double format (range [0,1])
% Output: seg - Binary boundary map (0s and 1s)

% Convert RGB to grayscale
if size(I,3) == 3
    Igray = rgb2gray(I);
else
    Igray = I;
end

% Parameters matching lab implementation
sigma = 3;
lambda = 0.1;
gamma = 0.75;

% Use multiple orientations for comprehensive edge detection
orientations = [0 45 90 135];
responses = zeros([size(Igray), length(orientations)]);

% Process each orientation
for i = 1:length(orientations)
    theta = orientations(i);
    
    % Generate Gabor filters with 90 and 0 degree phase shift
    gaborFilter90 = gabor2(sigma, lambda, theta, gamma, 90);
    gaborFilter0 = gabor2(sigma, lambda, theta, gamma, 0);
    
    % Convolve image with both filters
    gaborResponse90 = conv2(Igray, gaborFilter90, 'same');
    gaborResponse0 = conv2(Igray, gaborFilter0, 'same');
    
    % Complex cell response for this orientation
    responses(:,:,i) = sqrt(gaborResponse90.^2 + gaborResponse0.^2);
end

% Combine responses from all orientations
combinedResponse = max(responses, [], 3);

% Normalize response to [0,1] range
combinedResponse = (combinedResponse - min(combinedResponse(:))) / ...
                  (max(combinedResponse(:)) - min(combinedResponse(:)));

% Threshold to create binary boundary map
threshold = graythresh(combinedResponse);
seg = combinedResponse > threshold;

% Clean up boundaries
seg = bwmorph(seg, 'thin', Inf);  % Thin to single-pixel width
seg = bwareaopen(seg, 20);        % Remove small segments

end

function gb=gabor2(sigma,freq,orient,aspect,phase)
% Implementation of 2D Gabor filter - exact implementation from labs
% Parameters:
% sigma  = standard deviation of Gaussian envelope
% freq   = frequency of sine wave
% orient = orientation from horizontal (degrees)
% aspect = aspect ratio of Gaussian envelope
% phase  = phase of sine wave (degrees)

sz = fix(7*sigma/max(0.2,aspect));
if mod(sz,2)==0, sz=sz+1; end

[x y] = meshgrid(-fix(sz/2):fix(sz/2));

% Rotation 
orient = (orient-90)*pi/180;
xDash = x*cos(orient) + y*sin(orient);
yDash = -x*sin(orient) + y*cos(orient);

phase = phase*pi/180;

gb = exp(-.5*((xDash.^2/sigma^2)+(aspect^2*yDash.^2/sigma^2))) .* ...
     cos(2*pi*xDash*freq + phase);

% Normalize filter
gb(gb>0) = gb(gb>0)./sum(sum(max(0,gb)));
gb(gb<0) = gb(gb<0)./sum(sum(max(0,-gb)));
end
