clear;
%load 'real-sim.mat';
%load 'rcv1_train.binary.mat';
%load 'a9a.mat';
load 'Adult.mat';% 0.3232888469179914
%load 'covtype.mat';
%% Parse Data
X = [ones(size(X, 1), 1) X];
[N, Dim] = size(X);
X = full(X');

%% Normalize Data
sum1 = 1./sqrt(sum(X.^2, 1));
if abs(sum1(1) - 1) > 10^(-10)
    X = X.*repmat(sum1, Dim, 1);
end
clear sum1;

%% Set Params
algorithm = 'Ada_SVRG'; % SGD / SVRG / Prox_SVRG / Katyusha
passes = 240;
% For two-level algorithm, loop stands for outter loop count,
% for SGD, loop stands for total loop count.
loop = int64(passes / 3);
model = 'logistic'; % least_square / svm / logistic
regularizer = 'L2'; % L1 N/A for Katyusha / SVRG
init_weight = zeros(Dim, 1);
lambda = 10^(-6);
L = (0.25 * max(sum(X.^2, 1)) + lambda); % For logistic regression
sigma = lambda; % For Katyusha
step_size = 1 / (5 * L);
is_sparse = issparse(X);

is_plot = false;

fprintf('Algorithm: %s\n', algorithm);
fprintf('Model: %s-%s\n', regularizer, model);

tic;
% Mode 1: last_iter--last_iter, Mode 2: aver_iter--aver_iter, Mode 3: aver_iter--last_iter
Mode = 1; % For SVRG / Prox_SVRG
stored_SVRG_LL = Interface(X, y, algorithm, model, regularizer, init_weight, lambda, L, step_size, loop, is_sparse, Mode, sigma);
time = toc;
fprintf('Time: %f seconds \n', time);
fprintf('%.16f \n', stored_SVRG_LL);

% tic;
% Mode = 3;
% stored_SVRG_AA = Interface(X, y, algorithm, model, regularizer, init_weight, lambda, L, step_size, loop, is_sparse, Mode, sigma);
% time = toc;
% fprintf('Time: %f seconds \n', time);
%
% tic;
% algorithm = 'Katyusha';
% Mode = 3;
% stored_SVRG_AL = Interface(X, y, algorithm, model, regularizer, init_weight, lambda, L, step_size, loop, is_sparse, Mode, sigma);
% time = toc;
% fprintf('Time: %f seconds \n', time);

% algorithm = 'Prox_SVRG';
%
% Mode = 1;
% stored_PSVRG_LL = Interface(X, y, algorithm, model, regularizer, init_weight, lambda, L, step_size, loop, is_sparse, Mode, sigma);
%
% Mode = 2;
% stored_PSVRG_AA = Interface(X, y, algorithm, model, regularizer, init_weight, lambda, L, step_size, loop, is_sparse, Mode, sigma);
%
% Mode = 3;
% stored_PSVRG_AL = Interface(X, y, algorithm, model, regularizer, init_weight, lambda, L, step_size, loop, is_sparse, Mode, sigma);

% algorithm = 'Katyusha';
% stored_Katyusha = Interface(X, y, algorithm, model, regularizer, init_weight, lambda, L, step_size, loop, is_sparse, Mode, sigma);

% X axis
x1 = 1 : passes / 3;

%% Plot
if (is_plot)
    fEvals = cell(3, 1);
    fVals = cell(3, 1);
    fEvals{1} = x1' * 2;
    fEvals{2} = x1' * 2;
    fEvals{3} = x1' * 2;
    % fEvals{4} = x1' * 2;
    % fEvals{5} = x1' * 2;
    % fEvals{6} = x1' * 2;
    smallest_F = min([min(stored_SVRG_LL), min(stored_SVRG_AA), min(stored_SVRG_AA)]) - (6e-10);
    fVals{1} = stored_SVRG_LL - smallest_F;
    fVals{2} = stored_SVRG_AA - smallest_F;
    fVals{3} = stored_SVRG_AL - smallest_F;
    % fVals{1} = stored_F - fstar;
    % fVals{2} = stored_SVRG_AA - fstar;
    % fVals{3} = stored_SVRG_AL - fstar;
    % fVals{4} = stored_PSVRG_LL - fstar;
    % fVals{5} = stored_PSVRG_AA - fstar;
    % fVals{6} = stored_PSVRG_AL - fstar;

    n = length(fVals);

    colors = colormap(lines(8)); colors = colors([1 2 3 4 5 6 7], :);
    lineStyle = cellstr(['-'; '-'; '-'; '-'; '-'; '-'; '-'; '-']);
    markers = cellstr(['s'; 'o'; 'p'; '*'; 'd'; 'x'; 'h'; '+']);
    markerSpacing = [3 3 3 5 4 3 3 3; 2 1 3 2 4 6 2 4]';
    names = cellstr(['SVRG_L_L '; 'SVRG_A_A '; 'SVRG_A_L ']);%; 'PSVRG_L_L'; 'PSVRG_A_A'; 'PSVRG_A_L']);

    options.legendLoc = 'NorthEast';
    options.logScale = 2;
    options.colors = colors;
    options.lineStyles = lineStyle(1:n);
    options.markers = markers(1:n);
    options.markerSize = 12;
    options.markerSpacing = markerSpacing;
    options.legendStr = names;
    options.legend = names;
    options.ylabel = 'Objective';
    options.xlabel = 'Passes through Data';
    options.labelLines = 1;
    options.labelRotate = 1;
    options.xlimits = [];
    options.ylimits = [];

    prettyPlot(fEvals,fVals,options); % (Thanks, Mark Schmidt)
end
