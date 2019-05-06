function visualize_fields(a_)
% Plot function for velocity fields
% The function reconstructs the velocity fields using the 9 chosen Fourier
% modes and their corresponding amplitudes
%
% Input:
%   a - A matrix of size (nTP, 9), where nTP is the number of time points
%
% The code has been used for the results in:
% "Predictions of turbulent shear flows using deep neural networks"
% P.A. Srinivasan, L. Guastoni, H. Azizpour, P. Schlatter, R. Vinuesa
% Physical Review Fluids (accepted)
%%

% Visualization start and end time
start_t = 404;
end_t = start_t;

% Size of the domain
Lx = 4*pi;
Lz = 2*pi;

A = 2*pi/Lx;
B = pi/2;
C = 2*pi/Lz;

%% Velocity fields
% Number of gridpoints in each direction
nx = 21;
ny = nx;
nz = nx;

% The gridpoints
xp = linspace(0,Lx,nx);
yp = linspace(-1,1,ny);
zp = linspace(0,Lz,nz);

% Initialize arrays for each component of the 9 modes
u1x = zeros(nx,ny,nz);
u2x = zeros(nx,ny,nz);
u6x = zeros(nx,ny,nz);
u7x = zeros(nx,ny,nz);
u8x = zeros(nx,ny,nz);
u9x = zeros(nx,ny,nz);
u3y = zeros(nx,ny,nz);
u8y = zeros(nx,ny,nz);
u3z = zeros(nx,ny,nz);
u4z = zeros(nx,ny,nz);
u5z = zeros(nx,ny,nz);
u6z = zeros(nx,ny,nz);
u7z = zeros(nx,ny,nz);
u8z = zeros(nx,ny,nz);

% The velocity components of the chosen Fourier modes
for i = 1:nx
  for j = 1:ny
    for k = 1:nz
        u1x(i,j,k) = 2^0.5*sin(B*yp(j));
        u2x(i,j,k) = 4/3^0.5*(cos(B*yp(j)))^2*cos(C*zp(k));
        u6x(i,j,k) = 4*sqrt(2/3/(A^2 + C^2))*(-C)*cos(A*xp(i))...
            *(cos(B*yp(j)))^2*sin(C*zp(k));
        u7x(i,j,k) = 2*sqrt(2/(A^2 + C^2))*C*sin(A*xp(i))...
            *sin(B*yp(j))*sin(C*zp(k));
        u8x(i,j,k) = 2*sqrt(2/(A^2 + C^2)/(4*A^2 + 4*C^2 + pi^2))...
            *(pi*A)*sin(A*xp(i))*sin(B*yp(j))*sin(C*zp(k));
        u9x(i,j,k) = 2^0.5*sin(3*B*yp(j));
        u3y(i,j,k) = 2/sqrt(4*C^2 + pi^2)*2*C*cos(B*yp(j))*cos(C*zp(k));
        u8y(i,j,k) = 2*sqrt(2/(A^2 + C^2)/(4*A^2 + 4*C^2 + pi^2))...
            *2*(A^2+C^2)*cos(A*xp(i))*cos(B*yp(j))*sin(C*zp(k));
        u3z(i,j,k) = 2/sqrt(4*C^2 + pi^2)*pi*sin(B*yp(j))*sin(C*zp(k));
        u4z(i,j,k) = 4/3^0.5*cos(A*xp(i))*(cos(B*yp(j)))^2;
        u5z(i,j,k) = 2*sin(A*xp(i))*sin(B*yp(j));
        u6z(i,j,k) = 4*sqrt(2/3/(A^2 + C^2))*A*sin(A*xp(i))...
            *(cos(B*yp(j)))^2*cos(C*zp(k));
        u7z(i,j,k) = 2*sqrt(2/(A^2 + C^2))*A*cos(A*xp(i))...
            *sin(B*yp(j))*cos(C*zp(k));
        u8z(i,j,k) = 2*sqrt(2/(A^2 + C^2)/(4*A^2 + 4*C^2 + pi^2))...
            *(-pi*C)*cos(A*xp(i))*sin(B*yp(j))*cos(C*zp(k));
    end
  end
end

%% Plot

figure
set(gcf,'Position',[0 0 600 400])

% The grid for z-y plane
[ZP1,YP1] = meshgrid(zp,yp);

% The grid for x-z plane
[XP2,ZP2] = meshgrid(xp,zp);

for ti = start_t:end_t
    a = a_(ti,:);
    
    % velocity components
    ux = a(1)*u1x + a(2)*u2x + a(6)*u6x + a(7)*u7x + a(8)*u8x + a(9)*u9x;
    uy = a(3)*u3y + a(8)*u8y;
    uz = a(3)*u3z + a(4)*u4z + a(5)*u5z + a(6)*u6z + a(7)*u7z + a(8)*u8z;
    
    % average the velocities in x-direction (downstream)
    ux_avg = reshape(mean(ux,1),ny,nz);
    uy_avg = reshape(mean(uy,1),ny,nz);
    uz_avg = reshape(mean(uz,1),ny,nz);

    % velocities at midplane (between the plates)
    ux_mid = reshape(ux(:,ceil(ny/2),:),nx,nz);
    uy_mid = reshape(uy(:,ceil(ny/2),:),nx,nz);
    uz_mid = reshape(uz(:,ceil(ny/2),:),nx,nz);
    
    % Plot 1 - Mean profile
    subplot(2,7,[1 2])
    plot(mean(ux_avg,2),yp,'LineWidth',1.0)
    axis([-1 1 -1 1])
    xlabel('u_x','FontWeight','bold')
    ylabel('y','FontWeight','bold')

    % Plot 2 - Downstream
    subplot(2,7,[4 7])
    contourf(ZP1,YP1,ux_avg,'edgecolor','none')
    colormap jet
    colorbar
    caxis([-0.5 0.5])
    hold on
    quiver(ZP1,YP1,uz_avg,uy_avg, 'k')
    hold off
    axis([0 Lz -1 1])
    xlabel('z','FontWeight','bold')
    ylabel('y','FontWeight','bold')
    vec_pos = get(get(gca, 'XLabel'), 'Position');
    set(get(gca, 'XLabel'), 'Position', vec_pos + [0 0.1 0]);
    
    % Midplane between the plates
    subplot(2,7,[11 14])
    contourf(XP2,ZP2,uy_mid','edgecolor','none')
    colorbar
    caxis([-0.2 0.2])
    hold on
    quiver(XP2,ZP2,ux_mid',uz_mid','k')
    hold off
    axis([0 Lx 0 Lz])
    xlabel('x','FontWeight','bold')
    ylabel('z','FontWeight','bold')

    drawnow

end
