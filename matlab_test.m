
clear;
load('rcv1_train.binary.mat');

%% Why Transpose? Why Add 1-vector?
X = [ones(size(X, 1), 1) X];
[N, Dim] = size(X);
X = X';

algorithm = 'SVRG';
model = 'logistic';
regularizer = 'L2';
init_weight = zeros(d, 1);
lambda = 1 / N;
%% For logistic regression
L = (0.25 * max(sum(X.^2, 1)) + lambda);
step_size = 1.0 / (5.0 * L);
%% For two-level algorithm, stands for outter loop count,
%% for SGD, stands for total loop count.
loop = int64(10);
is_store_iterates = false;
is_sparse = issparse(X);

if (is_store_iterates)
    stored_weights = Interface(X, y, algorithm, model, regularizer, init_weight,
                            lambda, L, step_size, loop, is_sparse);
else
    Interface(X, y, algorithm, model, regularizer, init_weight, lambda, L, step_size,
                            loop, is_sparse);
end

%% Plot
if (is_store_iterates)


end
