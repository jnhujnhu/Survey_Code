
clear;
load('rcv1_train.binary.mat');

%% Transpose? Add 1-vector?
X = [ones(size(X, 1), 1) X];
[N, Dim] = size(X);
X = X';

algorithm = 'SGD';
model = 'logistic';
regularizer = 'L2';
init_weight = zeros(Dim, 1);
lambda = 1 / N;
%% For logistic regression
L = (0.25 * max(sum(X.^2, 1)) + lambda);
step_size = 1.0 / (5.0 * L);
%% For two-level algorithm, stands for outter loop count,
%% for SGD, stands for total loop count.
loop = int64(30 * N);
is_store_iterates = true;
is_sparse = issparse(X);

disp('Algorithm: ' + algorithm);
disp('Model: ' + regularizer + '-' + model);

if (is_store_iterates)
    stored_F = Interface(X, y, algorithm, model, regularizer, init_weight, lambda, L, step_size, loop, is_sparse);
    disp(stored_F)
else
    Interface(X, y, algorithm, model, regularizer, init_weight, lambda, L, step_size, loop, is_sparse);
end

%% Plot
if (is_store_iterates)


end
