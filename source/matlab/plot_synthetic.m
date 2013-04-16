%% Set random seed

rand('twister', 0);
randn('state', 0);

%% Linear trend

xdata = linspace(1, 5, 100);
ydata = xdata * 2 - 1;
ydata = ydata + randn(size(ydata)) * 0.5;

figure(1); clf; hold on;

% Figure settings.
lw = 1.2;
opacity = 1;
light_blue = [227 237 255]./255;

%set(gca,'Layer','top');  % Stop axes from being overridden.

plot( xdata, ydata, 'k.');

% Make plot prettier.
set(gcf, 'color', 'white');
set(gca, 'TickDir', 'out');

set_fig_units_cm( 16,8 );

%saveas( gcf, 'linear.pdf' );
save2pdf('linear.pdf', gcf, 600);

%% Quadratic trend

xdata = linspace(1, 5, 100);
ydata = -0.5 * xdata .* xdata + xdata * 2 - 1;
ydata = ydata + randn(size(ydata)) * 0.5;

figure(2); clf; hold on;

% Figure settings.
lw = 1.2;
opacity = 1;
light_blue = [227 237 255]./255;

%set(gca,'Layer','top');  % Stop axes from being overridden.

plot( xdata, ydata, 'k.');

% Make plot prettier.
set(gcf, 'color', 'white');
set(gca, 'TickDir', 'out');

set_fig_units_cm( 16,8 );

%saveas( gcf, 'linear.pdf' );
save2pdf('quadratic.pdf', gcf, 600);

%% Periodic trend

xdata = linspace(1, 5, 100);
ydata = 2 * sin(xdata * 2) + 1 * sin(xdata * 4) + xdata * 2 - 1;
ydata = ydata + randn(size(ydata)) * 0.25;

figure(3); clf; hold on;

% Figure settings.
lw = 1.2;
opacity = 1;
light_blue = [227 237 255]./255;

%set(gca,'Layer','top');  % Stop axes from being overridden.

plot( xdata, ydata, 'k.');

% Make plot prettier.
set(gcf, 'color', 'white');
set(gca, 'TickDir', 'out');

set_fig_units_cm( 16,8 );

%saveas( gcf, 'linear.pdf' );
save2pdf('periodic.pdf', gcf, 600);

%% Solar

load ../../data/1d_data/02-solar.mat

xdata = X;
ydata = y;

figure(4); clf; hold on;

% Figure settings.
lw = 1.2;
opacity = 1;
light_blue = [227 237 255]./255;

%set(gca,'Layer','top');  % Stop axes from being overridden.

plot( xdata, ydata, 'k.');

% Make plot prettier.
set(gcf, 'color', 'white');
set(gca, 'TickDir', 'out');

set_fig_units_cm( 16,8 );

%saveas( gcf, 'linear.pdf' );
save2pdf('solar.pdf', gcf, 600);

%% Airline

load ../../data/1d_data/01-airline.mat

xdata = X;
ydata = y;

figure(5); clf; hold on;

% Figure settings.
lw = 1.2;
opacity = 1;
light_blue = [227 237 255]./255;

%set(gca,'Layer','top');  % Stop axes from being overridden.

plot( xdata, ydata, 'k.');

% Make plot prettier.
set(gcf, 'color', 'white');
set(gca, 'TickDir', 'out');

set_fig_units_cm( 16,8 );

%saveas( gcf, 'linear.pdf' );
save2pdf('airline.pdf', gcf, 600);