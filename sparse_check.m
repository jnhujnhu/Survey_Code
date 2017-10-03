clear;
%load 'real-sim.mat';
%load 'rcv1_train.binary.mat';
load 'a9a.mat';
%load 'covtype.mat';
%% Parse Data
% X = [ones(size(X, 1), 1) X];
[N, Dim] = size(X);
X = X';

%% Normalize Data
% sum1 = 1./sqrt(sum(X.^2, 1));
% if abs(sum1(1) - 1) > 10^(-10)
%     X = X.*repmat(sum1, Dim, 1);
% end
% clear sum1;

%% Set Params
passes = 300;
model = 'least_square'; % least_square / svm / logistic
regularizer = 'elastic_net'; % L1 / L2 / elastic_net
init_weight = repmat(0, Dim, 1); % Initial weight
lambda1 = 10^(-4); % L2_norm / elastic_net
lambda2 = 10^(-5); % L1_norm / elastic_net
L = (max(sum(X.^2, 1)) + lambda1); % For ridge regression
sigma = lambda1; % For Katyusha / SAGA, Strong Convex Parameter
is_sparse = issparse(X);
Mode = 1;
is_plot = true;
fprintf('Model: %s-%s\n', regularizer, model);

% SVRG
algorithm = 'Prox_SVRG';
Mode = 3;
step_size = 1 / (20 * L);
loop = int64(passes / 3); % 3 passes per loop
fprintf('Algorithm: %s\n', algorithm);
tic;
hist1 = Interface(X, y, algorithm, model, regularizer, init_weight, lambda1, L, step_size, loop, is_sparse, Mode, sigma, lambda2, 0, 0, 0);
time = toc;
fprintf('Time: %f seconds \n', time);
X_SVRG = [0:3:passes]';
hist1 = [X_SVRG, hist1];

step_size = 1 / (15 * L);
fprintf('Algorithm: %s\n', algorithm);
tic;
hist2 = Interface(X, y, algorithm, model, regularizer, init_weight, lambda1, L, step_size, loop, is_sparse, Mode, sigma, lambda2, 0, 0, 0);
time = toc;
fprintf('Time: %f seconds \n', time);
hist2 = [X_SVRG, hist2];

step_size = 1 / (10 * L);
fprintf('Algorithm: %s\n', algorithm);
tic;
hist3 = Interface(X, y, algorithm, model, regularizer, init_weight, lambda1, L, step_size, loop, is_sparse, Mode, sigma, lambda2, 0, 0, 0);
time = toc;
fprintf('Time: %f seconds \n', time);
hist3 = [X_SVRG, hist3];

step_size = 1 / (5 * L);
fprintf('Algorithm: %s\n', algorithm);
tic;
hist4 = Interface(X, y, algorithm, model, regularizer, init_weight, lambda1, L, step_size, loop, is_sparse, Mode, sigma, lambda2, 0, 0, 0);
time = toc;
fprintf('Time: %f seconds \n', time);
hist4 = [X_SVRG, hist4];

step_size = 1 / (3 * L);
fprintf('Algorithm: %s\n', algorithm);
tic;
hist5 = Interface(X, y, algorithm, model, regularizer, init_weight, lambda1, L, step_size, loop, is_sparse, Mode, sigma, lambda2, 0, 0, 0);
time = toc;
fprintf('Time: %f seconds \n', time);
hist5 = [X_SVRG, hist5];

step_size = 1 / (2 * L);
fprintf('Algorithm: %s\n', algorithm);
tic;
hist6 = Interface(X, y, algorithm, model, regularizer, init_weight, lambda1, L, step_size, loop, is_sparse, Mode, sigma, lambda2, 0, 0, 0);
time = toc;
fprintf('Time: %f seconds \n', time);
hist6 = [X_SVRG, hist6];

step_size = 1 / (1.5 * L);
fprintf('Algorithm: %s\n', algorithm);
tic;
hist7 = Interface(X, y, algorithm, model, regularizer, init_weight, lambda1, L, step_size, loop, is_sparse, Mode, sigma, lambda2, 0, 0, 0);
time = toc;
fprintf('Time: %f seconds \n', time);
hist7 = [X_SVRG, hist7];

step_size = 1 / (1.25 * L);
fprintf('Algorithm: %s\n', algorithm);
tic;
hist8 = Interface(X, y, algorithm, model, regularizer, init_weight, lambda1, L, step_size, loop, is_sparse, Mode, sigma, lambda2, 0, 0, 0);
time = toc;
fprintf('Time: %f seconds \n', time);
hist8 = [X_SVRG, hist8];

clear X;
clear y;
load 'a9a.mat';
[N, Dim] = size(X);
X = full(X');
is_sparse = issparse(X);

step_size = 1 / (15 * L);
fprintf('Algorithm: %s\n', algorithm);
tic;
hist9 = Interface(X, y, algorithm, model, regularizer, init_weight, lambda1, L, step_size, loop, is_sparse, Mode, sigma, lambda2, 0, 0, 0);
time = toc;
fprintf('Time: %f seconds \n', time);
hist9 = [X_SVRG, hist9];

step_size = 1 / (10 * L);
fprintf('Algorithm: %s\n', algorithm);
tic;
hist10 = Interface(X, y, algorithm, model, regularizer, init_weight, lambda1, L, step_size, loop, is_sparse, Mode, sigma, lambda2, 0, 0, 0);
time = toc;
fprintf('Time: %f seconds \n', time);
hist10 = [X_SVRG, hist10];

step_size = 1 / (5 * L);
fprintf('Algorithm: %s\n', algorithm);
tic;
hist11 = Interface(X, y, algorithm, model, regularizer, init_weight, lambda1, L, step_size, loop, is_sparse, Mode, sigma, lambda2, 0, 0, 0);
time = toc;
fprintf('Time: %f seconds \n', time);
hist11 = [X_SVRG, hist11];

step_size = 1 / (3 * L);
fprintf('Algorithm: %s\n', algorithm);
tic;
hist12 = Interface(X, y, algorithm, model, regularizer, init_weight, lambda1, L, step_size, loop, is_sparse, Mode, sigma, lambda2, 0, 0, 0);
time = toc;
fprintf('Time: %f seconds \n', time);
hist12 = [X_SVRG, hist12];

step_size = 1 / (2 * L);
fprintf('Algorithm: %s\n', algorithm);
tic;
hist13 = Interface(X, y, algorithm, model, regularizer, init_weight, lambda1, L, step_size, loop, is_sparse, Mode, sigma, lambda2, 0, 0, 0);
time = toc;
fprintf('Time: %f seconds \n', time);
hist13 = [X_SVRG, hist13];

step_size = 1 / (1.5 * L);
fprintf('Algorithm: %s\n', algorithm);
tic;
hist14 = Interface(X, y, algorithm, model, regularizer, init_weight, lambda1, L, step_size, loop, is_sparse, Mode, sigma, lambda2, 0, 0, 0);
time = toc;
fprintf('Time: %f seconds \n', time);
hist14 = [X_SVRG, hist14];

step_size = 1 / (1.25 * L);
fprintf('Algorithm: %s\n', algorithm);
tic;
hist15 = Interface(X, y, algorithm, model, regularizer, init_weight, lambda1, L, step_size, loop, is_sparse, Mode, sigma, lambda2, 0, 0, 0);
time = toc;
fprintf('Time: %f seconds \n', time);
hist15 = [X_SVRG, hist15];

step_size = 1 / (20 * L);
fprintf('Algorithm: %s\n', algorithm);
tic;
hist16 = Interface(X, y, algorithm, model, regularizer, init_weight, lambda1, L, step_size, loop, is_sparse, Mode, sigma, lambda2, 0, 0, 0);
time = toc;
fprintf('Time: %f seconds \n', time);
hist16 = [X_SVRG, hist16];

%% Plot
if(is_plot)
    aa1 = min(hist1(:, 2));
    aa2 = min(hist2(:, 2));
    aa3 = min(hist3(:, 2));
    aa4 = min(hist4(:, 2));
    aa5 = min(hist5(:, 2));
    minval = min([aa2, aa4, aa5]) - 2e-16;
    aa = max(max([hist5(:, 2)])) - minval;
    b = 1;

    figure(101);
    set(gcf,'position',[200,100,386,269]);
    semilogy(hist1(1:b:end,1), abs(hist1(1:b:end,2) - minval),'b--o','linewidth',1.6,'markersize',4.5);
    hold on,semilogy(hist2(1:b:end,1), abs(hist2(1:b:end,2) - minval),'b--o','linewidth',1.6,'markersize',4.5);
    hold on,semilogy(hist3(1:b:end,1), abs(hist3(1:b:end,2) - minval),'b--o','linewidth',1.6,'markersize',4.5);
    hold on,semilogy(hist4(1:b:end,1), abs(hist4(1:b:end,2) - minval),'b--o','linewidth',1.6,'markersize',4.5);
    hold on,semilogy(hist5(1:b:end,1), abs(hist5(1:b:end,2) - minval),'b--o','linewidth',1.6,'markersize',4.5);
    hold on,semilogy(hist6(1:b:end,1), abs(hist6(1:b:end,2) - minval),'b--o','linewidth',1.6,'markersize',4.5);
    hold on,semilogy(hist7(1:b:end,1), abs(hist7(1:b:end,2) - minval),'b--o','linewidth',1.6,'markersize',4.5);
    hold on,semilogy(hist8(1:b:end,1), abs(hist8(1:b:end,2) - minval),'b--o','linewidth',1.6,'markersize',4.5);
    % hold on,semilogy(hist2(1:b:end,1), abs(hist2(1:b:end,2) - minval),'g-.^','linewidth',1.6,'markersize',4.5);
    % hold on,semilogy(hist3(1:b:end,1), abs(hist3(1:b:end,2) - minval),'c--+','linewidth',1.2,'markersize',4.5);
    hold on,semilogy(hist16(1:b:end,1), abs(hist16(1:b:end,2) - minval),'r-.d','linewidth',1.2,'markersize',4.5);
    hold on,semilogy(hist9(1:b:end,1), abs(hist9(1:b:end,2) - minval),'r-.d','linewidth',1.2,'markersize',4.5);
    hold on,semilogy(hist10(1:b:end,1), abs(hist10(1:b:end,2) - minval),'r-.d','linewidth',1.2,'markersize',4.5);
    hold on,semilogy(hist11(1:b:end,1), abs(hist11(1:b:end,2) - minval),'r-.d','linewidth',1.2,'markersize',4.5);
    hold on,semilogy(hist12(1:b:end,1), abs(hist12(1:b:end,2) - minval),'r-.d','linewidth',1.2,'markersize',4.5);
    hold on,semilogy(hist13(1:b:end,1), abs(hist13(1:b:end,2) - minval),'r-.d','linewidth',1.2,'markersize',4.5);
    hold on,semilogy(hist14(1:b:end,1), abs(hist14(1:b:end,2) - minval),'r-.d','linewidth',1.2,'markersize',4.5);
    hold on,semilogy(hist15(1:b:end,1), abs(hist15(1:b:end,2) - minval),'r-.d','linewidth',1.2,'markersize',4.5);
    % hold on,semilogy(hist6(1:b:end,1), abs(hist6(1:b:end,2) - minval),'k--<','linewidth',1.2,'markersize',4.5);
    hold off;
    xlabel('Number of effective passes');
    ylabel('Objective minus best');
    axis([0 50, 1E-12,aa]);
    legend('1/20', '1/15', '1/10', '1/5', '1/3', '1/2', '1/1.5', '1/1.25','1/20', '1/15', '1/10', '1/5', '1/3', '1/2', '1/1.5', '1/1.25'); %, 'SVRG', 'Prox-SVRG', 'Katyusha', 'SVRG-SD');
end
