function plot_amplitudes(a)
% Plot function for 9 the time series
% Input:
%   a - A matrix of size (nTP, 9), where nTP is the number of time points
%
% The code has been used for the results in:
% "Predictions of turbulent shear flows using deep neural networks"
% P.A. Srinivasan, L. Guastoni, H. Azizpour, P. Schlatter, R. Vinuesa
% Physical Review Fluids (accepted)
%%

% Time vector
t = 0:size(a,1)-1;

h = figure;
set(gcf,'Position',[0 0 800 600])



subplot(5,2,1)
plot(t,a(:,1))
axis([t(1) t(end) 0 1.1])
get(gca, 'XTick');
set(gca, 'FontSize', 8)
ylabel('$a_1$','Rotation',0, 'Interpreter','latex',...
    'FontSize',14)
vec_pos = get(get(gca, 'XLabel'), 'Position');
set(get(gca, 'XLabel'), 'Position', vec_pos + 1e3*[2.1 5.5e-4 0]);
vec_pos = get(get(gca, 'YLabel'), 'Position');
set(get(gca, 'YLabel'), 'Position', vec_pos + [-150 0.1 0]);



subplot(5,2,2)
plot(t,a(:,6))
axis([t(1) t(end) -0.45 0.45])
get(gca, 'XTick');
set(gca, 'FontSize', 8)
ylabel('$a_6$','Rotation',0, 'Interpreter','latex',...
    'FontSize',14)
vec_pos = get(get(gca, 'XLabel'), 'Position');
set(get(gca, 'XLabel'), 'Position', vec_pos + 1e3*[2.1 4.5e-4 0]);
vec_pos = get(get(gca, 'YLabel'), 'Position');
set(get(gca, 'YLabel'), 'Position', vec_pos + [-150 0.1 0]);



subplot(5,2,3)
plot(t,a(:,2))
axis([t(1) t(end) -0.45 0.45])
get(gca, 'XTick');
set(gca, 'FontSize', 8)
ylabel('$a_2$','Rotation',0, 'Interpreter','latex',...
    'FontSize',14)
vec_pos = get(get(gca, 'XLabel'), 'Position');
set(get(gca, 'XLabel'), 'Position', vec_pos + 1e3*[2.1 4.5e-4 0]);
vec_pos = get(get(gca, 'YLabel'), 'Position');
set(get(gca, 'YLabel'), 'Position', vec_pos + [-150 0.1 0]);



subplot(5,2,4)
plot(t,a(:,7))
axis([t(1) t(end) -0.45 0.45])
get(gca, 'XTick');
set(gca, 'FontSize', 8)
ylabel('$a_7$','Rotation',0, 'Interpreter','latex',...
    'FontSize',14)
vec_pos = get(get(gca, 'XLabel'), 'Position');
set(get(gca, 'XLabel'), 'Position', vec_pos + 1e3*[2.1 4.5e-4 0]);
vec_pos = get(get(gca, 'YLabel'), 'Position');
set(get(gca, 'YLabel'), 'Position', vec_pos + [-150 0.1 0]);



subplot(5,2,5)
plot(t,a(:,3))
axis([t(1) t(end) -0.45 0.45])
get(gca, 'XTick');
set(gca, 'FontSize', 8)
ylabel('$a_3$','Rotation',0, 'Interpreter','latex',...
    'FontSize',14)
vec_pos = get(get(gca, 'XLabel'), 'Position');
set(get(gca, 'XLabel'), 'Position', vec_pos + 1e3*[2.1 4.5e-4 0]);
vec_pos = get(get(gca, 'YLabel'), 'Position');
set(get(gca, 'YLabel'), 'Position', vec_pos + [-150 0.1 0]);



subplot(5,2,6)
plot(t,a(:,8))
axis([t(1) t(end) -0.45 0.45])
get(gca, 'XTick');
set(gca, 'FontSize', 8)
ylabel('$a_8$','Rotation',0, 'Interpreter','latex',...
    'FontSize',14)
vec_pos = get(get(gca, 'XLabel'), 'Position');
set(get(gca, 'XLabel'), 'Position', vec_pos + 1e3*[2.1 4.5e-4 0]);
vec_pos = get(get(gca, 'YLabel'), 'Position');
set(get(gca, 'YLabel'), 'Position', vec_pos + [-150 0.1 0]);



subplot(5,2,7)
plot(t,a(:,4))
axis([t(1) t(end) -0.45 0.45])
get(gca, 'XTick');
set(gca, 'FontSize', 8)
ylabel('$a_4$','Rotation',0, 'Interpreter','latex',...
    'FontSize',14)
vec_pos = get(get(gca, 'XLabel'), 'Position');
set(get(gca, 'XLabel'), 'Position', vec_pos + 1e3*[2.1 4.5e-4 0]);
vec_pos = get(get(gca, 'YLabel'), 'Position');
set(get(gca, 'YLabel'), 'Position', vec_pos + [-150 0.1 0]);



subplot(5,2,8)
plot(t,a(:,9))
axis([t(1) t(end) -0.5 0.05])
get(gca, 'XTick');
set(gca, 'FontSize', 8)
xlabel('$t$', 'Interpreter','latex','FontSize',14)
ylabel('$a_9$','Rotation',0, 'Interpreter','latex',...
    'FontSize',14)
vec_pos = get(get(gca, 'YLabel'), 'Position');
set(get(gca, 'YLabel'), 'Position', vec_pos + [-150 0.1 0]);



subplot(5,2,9)
plot(t,a(:,5))
axis([t(1) t(end) -0.45 0.45])
get(gca, 'XTick');
set(gca, 'FontSize', 8)
xlabel('$t$', 'Interpreter','latex','FontSize',14)
ylabel('$a_5$','Rotation',0, 'Interpreter','latex',...
    'FontSize',14)
vec_pos = get(get(gca, 'YLabel'), 'Position');
set(get(gca, 'YLabel'), 'Position', vec_pos + [-150 0.1 0]);



% set(h,'Units','Inches');
% pos = get(h,'Position');
% set(h,'PaperPositionMode','Auto','PaperUnits','Inches',...
%     'PaperSize',[pos(3), pos(4)])
% print('-bestfit',h,'test','-dpdf','-r0')

end
