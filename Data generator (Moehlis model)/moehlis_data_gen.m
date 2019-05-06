%% Moehlis data generator
% This file generates multiple time series and saves them in a single file.
% These time series will be used as training data for neural networks.
% 
% Output:
%   moehlis_data_###.mat
%
% The code has been used for the results in:
% "Predictions of turbulent shear flows using deep neural networks"
% P.A. Srinivasan, L. Guastoni, H. Azizpour, P. Schlatter, R. Vinuesa
% Physical Review Fluids (accepted)
%%

% Number of time series in the output file
nTS = 10;

% Number of timepoints
nTP = 4000;

% Time interval between the timepoints
dt = 1;

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

%%
% Initialize empty 3D matrix for storing data
data = zeros(nTS, nTP, 9);

% Initial conditions
init = [1 0.07066 -0.07076 0 0 0 0 0 0];

count = 1;
while count <= nTS
    disp(count)

    % Add a random perturbation to init(4)
    init(4) = 0.1*rand;

    % Solve ODE
    [t,a_] = ode15s(@(t,a) moehlis_model_odefun(t,a,Re), 0:dt:nTP*dt+99, init);

    % Take only the last nTP points
    a_ = a_(end-nTP+1:end, :);

    % Check for laminarization and add to data matrix only if not
    ind = find(abs(a_(:,1)-1) < 0.01, 1);
    if isempty(ind)
        data(count,:,:) = a_;
        count = count + 1;
    end
end

save(['./moehlis_data_' num2str(nTS) '.mat'], 'data')
