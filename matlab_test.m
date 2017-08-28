
clear;
load('rcv1_train.binary.mat');

%% Transpose? Add 1-vector?
X = [ones(size(X, 1), 1) X];
[N, Dim] = size(X);
X = X';

algorithm = 'SVRG';
model = 'logistic';
regularizer = 'L2';
init_weight = zeros(Dim, 1);
lambda = 1 / N;
%% For logistic regression
L = (0.25 * max(sum(X.^2, 1)) + lambda);
step_size = 1.0 / (5.0 * L);
%% For two-level algorithm, stands for outter loop count,
%% for SGD, stands for total loop count.
passes = 30;
loop = int64(passes / 2);
is_store_iterates = true;
is_plot = false;
is_sparse = issparse(X);

fprintf('Algorithm: %s\n', algorithm);
fprintf('Model: %s-%s\n', regularizer, model);
if (is_store_iterates)
    stored_F = Interface(X, y, algorithm, model, regularizer, init_weight, lambda, L, step_size, loop, is_sparse);
    disp(stored_F)
else
    Interface(X, y, algorithm, model, regularizer, init_weight, lambda, L, step_size, loop, is_sparse);
end

% algorithm = 'SVRG';
% loop = int64(passes / 2);
%
% fprintf('Algorithm: %s\n', algorithm);
% fprintf('Model: %s-%s\n', regularizer, model);
% if (is_store_iterates)
%     stored_SVRG = Interface(X, y, algorithm, model, regularizer, init_weight, lambda, L, step_size, loop, is_sparse);
%     disp(stored_SVRG)
% else
%     Interface(X, y, algorithm, model, regularizer, init_weight, lambda, L, step_size, loop, is_sparse);
% end

% X axis
x1 = 1 : passes;
temp = ones(passes, 1) - 0.8;


%% Plot
if (is_plot && is_store_iterates)
    fEvals = cell(4, 1);
    fVals = cell(4, 1);
    fEvals{1} = x1';
    fEvals{2} = x1';
    fEvals{3} = x1';
    fEvals{4} = x1';
    fVals{1} = stored_F;
    fVals{2} = stored_SVRG;
    fVals{3} = stored_Ktyus;
    fVals{4} = temp;

    n = length(fVals);

    colors = colormap(lines(8)); colors = colors([1 2 3 5], :);
    lineStyle = cellstr(['-'; '-'; '-'; '-'; '-'; '-'; '-'; '-']);
    markers = cellstr(['s'; 'o'; 'p'; '*'; 'd'; 'x'; 'h'; '+']);
    markerSpacing = [3 3 3 5 4 3 3 3; 2 1 3 2 4 6 2 4]';
    names = cellstr(['SGD  '; 'SVRG '; 'Ktyus'; 'BASE ']);

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
