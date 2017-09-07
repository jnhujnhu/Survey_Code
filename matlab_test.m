clear;
%load 'real-sim.mat';
%load 'rcv1_train.binary.mat';
%load 'a9a.mat';
load 'Adult.mat';% 0.3232888469179914
%load 'covtype.mat';
%% Parse Data
X = [ones(size(X, 1), 1) X];
[N, Dim] = size(X);
X = X';

%% Normalize Data
sum1 = 1./sqrt(sum(X.^2, 1));
if abs(sum1(1) - 1) > 10^(-10)
    X = X.*repmat(sum1, Dim, 1);
end
clear sum1;

%% Set Params
algorithm = 'SAGA'; % SGD / SVRG / Prox_SVRG / Katyusha
passes = 240;
% For two-level algorithm, loop stands for outter loop count,
% for SGD, loop stands for total loop count.
loop = int64(passes * N);
model = 'logistic'; % least_square / svm / logistic
regularizer = 'L2'; % L1 N/A for Katyusha / SVRG
init_weight = repmat(0, Dim, 1);
lambda = 10^(-6);
L = (0.25 * max(sum(X.^2, 1)) + lambda); % For logistic regression
sigma = lambda; % For Katyusha
step_size = 1 / (3 *(sigma * N + L));
is_sparse = issparse(X);

is_plot = true;

fprintf('Algorithm: %s\n', algorithm);
fprintf('Model: %s-%s\n', regularizer, model);

tic;
% Mode 1: last_iter--last_iter, Mode 2: aver_iter--aver_iter, Mode 3: aver_iter--last_iter
Mode = 1; % For SVRG / Prox_SVRG
stored_SAGA_s = Interface(X, y, algorithm, model, regularizer, init_weight, lambda, L, step_size, loop, is_sparse, Mode, sigma);
stored_1 = stored_SAGA_s(3:3:end);
time = toc;
fprintf('Time: %f seconds \n', time);
% fprintf('%.16f \n', stored_1);

algorithm = 'SVRG';
step_size = 1 / (5 * L);
loop = int64(passes / 3);
tic;
Mode = 1;
stored_2 = Interface(X, y, algorithm, model, regularizer, init_weight, lambda, L, step_size, loop, is_sparse, Mode, sigma);
time = toc;
fprintf('Time: %f seconds \n', time);

tic;
algorithm = 'Katyusha';
Mode = 3;
stored_3 = Interface(X, y, algorithm, model, regularizer, init_weight, lambda, L, step_size, loop, is_sparse, Mode, sigma);
time = toc;
fprintf('Time: %f seconds \n', time);

% X axis
x1 = 1 : passes / 3;

%% Plot
if (is_plot)
    fEvals = cell(3, 1);
    fVals = cell(3, 1);
    fEvals{1} = x1' * 2;
    fEvals{2} = x1' * 2;
    fEvals{3} = x1' * 2;

    smallest_F = min([min(stored_1), min(stored_2), min(stored_3)]) - (6e-17);
    fVals{1} = stored_1 - smallest_F;
    fVals{2} = stored_2 - smallest_F;
    fVals{3} = stored_3 - smallest_F;

    n = length(fVals);

    colors = colormap(lines(8)); colors = colors([1 2 3 4 5 6 7], :);
    lineStyle = cellstr(['-'; '-'; '-'; '-'; '-'; '-'; '-'; '-']);
    markers = cellstr(['s'; 'o'; 'p'; '*'; 'd'; 'x'; 'h'; '+']);
    markerSpacing = [3 3 3 5 4 3 3 3; 2 1 3 2 4 6 2 4]';
    names = cellstr(['SAGA '; 'SVRG '; 'KATYU']);%; 'PSVRG_L_L'; 'PSVRG_A_A'; 'PSVRG_A_L']);

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
