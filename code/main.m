% This is the main code

clear all
close all
clc

% Defining parameters
% Filtering options
filtOpts.minSVD = 5;
filtOpts.maxSVD = 500; 
filtOpts.psfX = 2; % [wvl]
filtOpts.psfZ = 1;
filtOpts.iter = 4;
filtOpts.noise = 0.00;

% Set peak finding options for Morphological reconstruction
pkOpts.offset = 0.25; 
pkOpts.threshold = 0.95; 
pkOpts.interpFactor = 1;
pkOpts.rgnSize = [0.5, 7.5].^2;
pkOpts.sigma = 1;
pkOpts.usePeak = false;
pkOpts.proxThreshold = 0.5; 


% Image options
imgOpts.interpFactor = 1;
imgOpts.pointDither = 0.25;
imgOpts.psf_wvl = 1;

DATA_FRACTION = [0.0, 0.01]; %set the desired portion of the dataset

% Get video files
vidFile = 'simuData.avi';

% Get relevant metadata files
metaFile = 'metadata.mat';

% Load metadata
load(metaFile);
px = PxSet; % Rename
xVec = px.mapX;
zVec = px.mapZ;
[x, z] = meshgrid( xVec, zVec );

imgFreq = SimSet.centre_frequency;
lambda = 1540./imgFreq;
Fs = SimSet.sampling_frequency;
dt = 1./Fs;

% Read video file
vid = VideoReader(vidFile);
video = read(vid);

% Convert to black and white
numFrames = size( video, 4 );
Nx = size( video, 2 );
Nz = size( video, 1 );
bwVideo = zeros( numFrames, Nz, Nx );
for fCount = 1 : numFrames
    bwFrame = rgb2gray(video(:,:,:,fCount));
    bwVideo( fCount, :, : ) = bwFrame;
end

% Add noise if required
if filtOpts.noise > 0
    nL = filtOpts.noise./2;
    bwVideo = ( 1 + nL.*randn(size(bwVideo)) ).*bwVideo;
end

% Keep only desired portion of data
totalFrames = size( bwVideo, 1 );
startFrame = ceil( max( DATA_FRACTION(1).*totalFrames, 1 ) );
endFrame = floor( min( DATA_FRACTION(2).*totalFrames, totalFrames ) );
bwVideo = bwVideo( startFrame : endFrame, :, : );
numFrames = endFrame - startFrame + 1;

% Set SVD Values
firstSvdVal = filtOpts.minSVD;
lastSvdVal = round( filtOpts.maxSVD.*(endFrame - startFrame)./totalFrames );
SVD_VALS = [firstSvdVal, lastSvdVal];

% SVD Filter
bwVideoRaw = bwVideo;
[bwVideo, Svals] = svdFilt( bwVideoRaw, SVD_VALS );

% Deconvolution
bwVideoSvd = abs(bwVideo);
if filtOpts.iter > 0

    bwVideoDeconv = 0.*bwVideo;

    sx = filtOpts.psfX.*lambda;
    sz = filtOpts.psfZ.*lambda;
    z0 = mean(zVec);
    psfDim = 1.5.*max( sx, sz );

    xLVec = -psfDim : px.dx : psfDim;
    zLVec = -psfDim : px.dz : psfDim;
    [xL, zL] = meshgrid( xLVec, zLVec );
    psf = exp( -( (xL./sx).^(2) + (zL./sz).^(2) ) );
    parfor fCount = 1 : numFrames
        frame = squeeze(bwVideo( fCount, :, : ));
        deconvFrame = deconvlucy( frame, psf, filtOpts.iter );
        bwVideoDeconv( fCount, :, : ) = deconvFrame;

    end
else
    bwVideoDeconv = bwVideo;
end

% If zero iterations were considered for the deconvolution, then use the
% SVD-filtered data only
if filtOpts.iter <= 0
    frameData = bwVideoSvd;
else
    frameData = bwVideoDeconv;
end

img = zeros( Nz, Nx, numFrames );

% Get peaks for each frame
for fCount = 1 : numFrames

    % Get peaks for that frame
    rawFrame = squeeze( bwVideoRaw( fCount, :, : ) );
    frame = squeeze( frameData( fCount, :, : ) );

    % Normalize each
    frame = ( frame - min(frame(:)) )./( max(frame(:)) - min(frame(:)) );

    [xPeaks, zPeaks, peakVals] = ...
        isolateBubbles(x, z, frame, pkOpts);

    sz = imgOpts.psf_wvl.*lambda;
    img(:,:, fCount) = makeSrImg(xPeaks, zPeaks, sz, x, z);

end

frameCount = input("Enter the frame for flow estimation: ");

lambda = 0.3;
% Call the optical_flow_TV function to compute the motion field
[u, v, frame1, frame2] = optical_flow_TV(img, frameCount, lambda, 10);

% Plot the original frame
imagesc(frame1);colormap('gray');

% Create a grid of x and y coordinates
[h, w] = size(frame1);
[x, y] = meshgrid(1:w, 1:h);

hold on
% Plot the velocity vectors on top of the original frame
quiver(x, y, u, v, 'r');

% Show the plot
title('Optical Flow');
drawnow;