
clear;
load('rcv1_train.binary.mat');

%% Parse Data
X = [ones(size(X, 1), 1) X];
[N, Dim] = size(X);
X = X';

%% Set Params
algorithm = 'SVRG'; % SGD / SVRG / Prox_SVRG / Katyusha
passes = 30;
% For two-level algorithm, loop stands for outter loop count,
% for SGD, loop stands for total loop count.
loop = int64(passes / 2);
model = 'logistic'; % least_square / svm / logistic
regularizer = 'L2'; % L1 N/A for Katyusha
init_weight = zeros(Dim, 1);
lambda = 1 / N;
% Mode 1: last_iter--last_iter, Mode 2: aver_iter--aver_iter, Mode 3: aver_iter--last_iter
Mode = 1; % For SVRG
%% For logistic regression
L = (0.25 * max(sum(X.^2, 1)) + lambda);
step_size = 1.0 / (5.0 * L);
is_store_iterates = true;
is_plot = false;
is_sparse = issparse(X);

fprintf('Algorithm: %s\n', algorithm);
fprintf('Model: %s-%s\n', regularizer, model);
if (is_store_iterates)
    stored_F = Interface(X, y, algorithm, model, regularizer, init_weight, lambda, L, step_size, loop, is_sparse, Mode);
    disp(stored_F);
else
    Interface(X, y, algorithm, model, regularizer, init_weight, lambda, L, step_size, loop, is_sparse, Mode);
end

% Mode = 2;
% stored_SVRG_AA = Interface(X, y, algorithm, model, regularizer, init_weight, lambda, L, step_size, loop, is_sparse, Mode);
%
% Mode = 3;
% stored_SVRG_AL = Interface(X, y, algorithm, model, regularizer, init_weight, lambda, L, step_size, loop, is_sparse, Mode);

% X axis
x1 = 1 : passes;
temp = ones(passes, 1) - 0.8;


%% Plot
fstar = 0.201831346413416; % for lambda = 1/n;
if (is_plot && is_store_iterates)
    fEvals = cell(3, 1);
    fVals = cell(3, 1);
    fEvals{1} = x1';
    fEvals{2} = x1';
    fEvals{3} = x1';
    fVals{1} = stored_F - fstar;
    fVals{2} = stored_SVRG_AA - fstar;
    fVals{3} = stored_SVRG_AL - fstar;

    n = length(fVals);

    colors = colormap(lines(8)); colors = colors([1 2 3], :);
    lineStyle = cellstr(['-'; '-'; '-'; '-'; '-'; '-'; '-'; '-']);
    markers = cellstr(['s'; 'o'; 'p'; '*'; 'd'; 'x'; 'h'; '+']);
    markerSpacing = [3 3 3 5 4 3 3 3; 2 1 3 2 4 6 2 4]';
    names = cellstr(['SVRG_L_L'; 'SVRG_A_A'; 'SVRG_A_L']);

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
