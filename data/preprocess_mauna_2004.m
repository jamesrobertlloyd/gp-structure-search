% This script preprocesses the mauna dataset to hopefully match the experiment
% from "Gaussian Processes for Machine Learning" page 119.
%
% Based off of
% http://pmtksupport.googlecode.com/svn/trunk/GPstuff-2.0/gp/demo_periodicCov.m
% and
% http://cdiac.esd.ornl.gov/ftp/trends/co2/maunaloa.co2
%
% David Duvenaud
% dkd23@Cam.ac.uk
% Nov 2012

clear;

data=load('maunaloa2004.txt');
y = data(:, 2:13);
y=y';
y=y(:);
X = [1:1:length(y)]';

% Remove non-measurements.
X = X(y>0);
y = y(y>0);
avgy = mean(y);
y = y-avgy;

% Remove the 2004 data, to match Carl's experiments which only use to the end of
% 2003.  At this point, there should be 545 datapoints.
X(end-11:end) = [];
y(end-11:end) = [];

assert(numel(X) == 545);
assert(numel(y) == 545);

% Rescale the X's to correspond to years.
X = (X - 1)./ 12 + 1958;

save ('mauna2003', 'X', 'y' );
