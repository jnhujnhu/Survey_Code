clear;
%load 'real-sim.mat';
%load 'rcv1_train.binary.mat';
load 'a9a.mat';
%load 'Covtype.mat';
%% Parse Data
% X = [ones(size(X, 1), 1) X];
[N, Dim] = size(X);
X = full(X');

%% Normalize Data
% sum1 = 1./sqrt(sum(X.^2, 1));
% if abs(sum1(1) - 1) > 10^(-10)
%     X = X.*repmat(sum1, Dim, 1);
% end
% clear sum1;

%% Set Params
passes = 150;
model = 'least_square'; % least_square / svm / logistic
regularizer = 'L2'; % L1 / L2 / elastic_net
init_weight = repmat(0, Dim, 1); % Initial weight
lambda1 = 10^(-5); % L2_norm / elastic_net
lambda2 = 10^(-5); % L1_norm / elastic_net
L = (max(sum(X.^2, 1)) + lambda1); % For logistic regression
sigma = lambda1;
is_sparse = issparse(X);
Mode = 1;
is_plot = true;
fprintf('Model: %s-%s\n', regularizer, model);

%% SVRG
algorithm = 'SVRG';
Mode = 1;
step_size = 4 / (5 * L);
loop = int64(passes / 3); % 3 passes per loop
fprintf('Algorithm: %s\n', algorithm);
tic;
hist1 = Interface(X, y, algorithm, model, regularizer, init_weight, lambda1, L, step_size, loop, is_sparse, Mode, sigma, lambda2);
time = toc;
fprintf('Time: %f seconds \n', time);
X_SVRG = [0:3:passes]';
hist1 = [X_SVRG, hist1];
r = Dim;
A = 0;

%% SVRG_SD
algorithm = 'SVRG_SD';
sigma = 1.0 / 3.0; % Momentum Constant
interval = 2000; % Sufficient Decrease Iterate Interval
step_size = 9.6 / (5 * L);
loop = int64(passes / 3); % 3 passes per loop
fprintf('Algorithm: %s\n', algorithm);
% for partial SVD(in dense case)
% SVD for dense case
if(~is_sparse)
    [U, S, V] = svd(X', 'econ');
    A = (S * V')';
end
tic;
hist2 = Interface(X, y, algorithm, model, regularizer, init_weight, lambda1, L, step_size, loop, is_sparse, Mode, sigma, lambda2, interval, r, A);
time = toc;
fprintf('Time: %f seconds \n', time);
hist2 = [X_SVRG, hist2];

%% SVRG_LS
% LSF_Mode: SVRG_LS_FULL = 4;
% LSF_Mode: SVRG_LS_STOC = 5;
%
% LSC_Mode: SVRG_LS_CHGF = 6;
% LSC_Mode: SVRG_LS_OUTF = 7;
%
% LSM_Mode: SVRG_LS_SVD = 8;
% LSM_Mode: SVRG_LS_A = 9;
LSF_Mode = 4;
LSC_Mode = 6;
LSM_Mode = 8;
algorithm = 'SVRG_LS';
Mode = 1;
step_size = 4 / (5 * L);
interval = 3000;
loop = int64(passes / 3); % 3 passes per loop
fprintf('Algorithm: %s\n', algorithm);
tic;
hist3 = Interface(X, y, algorithm, model, regularizer, init_weight, lambda1, L, step_size, loop, is_sparse, Mode, sigma, lambda2, interval, r, A, LSF_Mode, LSC_Mode, LSM_Mode);
time = toc;
fprintf('Time: %f seconds \n', time);
hist3 = [X_SVRG, hist3];
clear X_SVRG;

%% Plot
if(is_plot)
    aa1 = min(hist1(:, 2));
    aa2 = min(hist2(:, 2));
    aa3 = min(hist3(:, 2));
    minval = min([aa1, aa2, aa3]) - 2e-16;
    aa = max(max([hist1(:, 2)])) - minval;
    b = 1;

    figure(101);
    set(gcf,'position',[200,100,386,269]);
    semilogy(hist1(1:b:end,1), abs(hist1(1:b:end,2) - minval),'b--o','linewidth',1.6,'markersize',4.5);
    hold on,semilogy(hist2(1:b:end,1), abs(hist2(1:b:end,2) - minval),'g-.^','linewidth',1.6,'markersize',4.5);
    hold on,semilogy(hist3(1:b:end,1), abs(hist3(1:b:end,2) - minval),'c--+','linewidth',1.2,'markersize',4.5);
    % hold on,semilogy(hist4(1:b:end,1), abs(hist4(1:b:end,2) - minval),'r-.d','linewidth',1.2,'markersize',4.5);
    % hold on,semilogy(hist5(1:b:end,1), abs(hist5(1:b:end,2) - minval),'k--<','linewidth',1.2,'markersize',4.5);
    % hold on,semilogy(hist6(1:b:end,1), abs(hist6(1:b:end,2) - minval),'m--<','linewidth',1.2,'markersize',4.5);
    hold off;
    xlabel('Number of effective passes');
    ylabel('Objective minus best');
    axis([0 100, 1E-12,aa]);
    legend('SVRG', 'SVRG-SD', 'SVRG-LS');
end
