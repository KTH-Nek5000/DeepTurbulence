%% Moehlis model Script
% This file generates a sample time series for the 9 amplitudes using the
% ODE model from Moehlis et al.
%
% The code has been used for the results in:
% "Predictions of turbulent shear flows using deep neural networks"
% P.A. Srinivasan, L. Guastoni, H. Azizpour, P. Schlatter, R. Vinuesa
% Physical Review Fluids (accepted)
%%

%% Parameters
% Reynolds number
Re = 400;

% Size of the domain
Lx = 4*pi;
Lz = 2*pi;

global A B C k1 k2 k3

A = 2*pi/Lx;
B = pi/2;
C = 2*pi/Lz;

k1 = sqrt(A^2 + C^2);
k2 = sqrt(B^2 + C^2);
k3 = sqrt(A^2 + B^2 + C^2);

%% To Solve the system of ODEs
% Number of timepoints
nTP = 4000;

% Time interval between the timepoints
dt = 1;

% Initial conditions
init = [1 0.07066 -0.07076 0 0 0 0 0 0];

% Add a perturbation to init(4)
init(4) = 1e-4;

% Solve ODE
[t,a_] = ode15s(@(t,a) moehlis_model_odefun(t,a,Re), 0:dt:nTP*dt+99, init);
t = t(1:nTP, :);
a_ = a_(end-nTP+1:end, :);
% The above line generates a time series until t=4100, and chops off the 
% first 100 points. This is done to eliminate the influence of the initial
% conditions on the resulting time series which serves as a more diverse
% data set.

% plot_amplitudes(a_)
visualize_fields(a_)
