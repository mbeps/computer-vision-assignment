function [seg] = segment_image(I)
% Image segmentation using both simple and complex cells
% Input: I - RGB image in double format (range [0,1])
% Output: seg - Binary boundary map (0s and 1s)

% Convert RGB to grayscale
if size(I,3) == 3
    Igray = rgb2gray(I);
else
    Igray = I;
end

% Parameters from lab implementation
sigma = 3;
lambda = 0.1;
gamma = 0.75;
orientations = [0 45 90 135];

% Initialize response matrices
complexResponses = zeros([size(Igray), length(orientations)]);
simpleResponses = zeros([size(Igray), length(orientations)]);

% Process each orientation
for i = 1:length(orientations)
    theta = orientations(i);
    
    % Simple cells (phase = 90)
    gaborFilterSimple = gabor2(sigma, lambda, theta, gamma, 90);
    simpleResponse = conv2(Igray, gaborFilterSimple, 'same');
    simpleResponses(:,:,i) = abs(simpleResponse);  % Take absolute value for edge strength
    
    % Complex cells (phase = 0 and 90)
    gaborFilter90 = gabor2(sigma, lambda, theta, gamma, 90);
    gaborFilter0 = gabor2(sigma, lambda, theta, gamma, 0);
    
    response90 = conv2(Igray, gaborFilter90, 'same');
    response0 = conv2(Igray, gaborFilter0, 'same');
    
    complexResponses(:,:,i) = sqrt(response90.^2 + response0.^2);
end

% Combine responses across orientations
maxSimpleResponse = max(simpleResponses, [], 3);
maxComplexResponse = max(complexResponses, [], 3);

% Combine simple and complex responses
% Weight complex cells more since they're better at general boundary detection
combinedResponse = 0.3 * maxSimpleResponse + 0.7 * maxComplexResponse;

% Normalize response to [0,1] range
combinedResponse = (combinedResponse - min(combinedResponse(:))) / ...
                  (max(combinedResponse(:)) - min(combinedResponse(:)));

% Apply non-maximum suppression to thin edges
[Gmag, Gdir] = imgradient(combinedResponse);
segedges = edge(combinedResponse, 'canny', [0.1 0.2]);

% Clean up boundaries
seg = bwmorph(segedges, 'thin', Inf);  % Ensure single-pixel width
seg = bwareaopen(seg, 20);             % Remove small segments

% Close small gaps
se = strel('disk', 1);
seg = imclose(seg, se);

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
