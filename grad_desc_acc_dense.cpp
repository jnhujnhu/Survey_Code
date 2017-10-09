#include "grad_desc_acc_dense.hpp"
#include "utils.hpp"
#include "regularizer.hpp"
#include <random>
#include <cmath>
#include <string.h>

extern size_t MAX_DIM;

// "Stochastic Proximal Gradient Descent with Acceleration Techniques"
std::vector<double>* grad_desc_acc_dense::Acc_Prox_SVRG1(double* X, double* Y, size_t N, blackbox* model
    , size_t iteration_no, double L, double sigma, double step_size, bool is_store_result) {
    // Random Generator
    std::random_device rd;
    std::default_random_engine generator(rd());
    std::uniform_int_distribution<int> distribution(0, N - 1);
    // std::vector<double>* stored_weights = new std::vector<double>;
    std::vector<double>* stored_F = new std::vector<double>;
    double* x = new double[MAX_DIM];
    double* y = new double[MAX_DIM];
    double* full_grad = new double[MAX_DIM];
    double m0 = (double) N * 2.0;
    int regular = model->get_regularizer();
    double* lambda = model->get_params();
    size_t total_iterations = 0;
    copy_vec(x, model->get_model());
    copy_vec(y, model->get_model());
    // Init Weight Evaluate
    if(is_store_result)
        stored_F->push_back(model->zero_oracle_dense(X, Y, N));
    // OUTTER_LOOP
    for(size_t i = 0 ; i < iteration_no; i ++) {
        double* full_grad_core = new double[N];
        double inner_m = m0;
        memset(full_grad, 0, MAX_DIM * sizeof(double));
        // Full Gradient
        for(size_t j = 0; j < N; j ++) {
            full_grad_core[j] = model->first_component_oracle_core_dense(X, Y, N, j);
            for(size_t k = 0; k < MAX_DIM; k ++) {
                full_grad[k] += X[j * MAX_DIM + k] * full_grad_core[j] / (double) N;
            }
        }
        copy_vec(x, model->get_model());
        copy_vec(y, model->get_model());
        // INNER_LOOP
        for(size_t j = 0; j < inner_m; j ++) {
            int rand_samp = distribution(generator);
            double inner_core = model->first_component_oracle_core_dense(X, Y, N
                , rand_samp, y);
            for(size_t k = 0; k < MAX_DIM; k ++) {
                double val = X[rand_samp * MAX_DIM + k];
                double vr_sub_grad = (inner_core - full_grad_core[rand_samp]) * val + full_grad[k];
                double prev_x = x[k];
                y[k] -= step_size * vr_sub_grad;
                x[k] = regularizer::proximal_operator(regular, y[k], step_size, lambda);
                y[k] = x[k] + sigma * (x[k] - prev_x);
            }
            total_iterations ++;
        }
        model->update_model(x);
        // For Matlab (per m/n passes)
        if(is_store_result) {
            stored_F->push_back(model->zero_oracle_dense(X, Y, N));
        }
        delete[] full_grad_core;
    }
    delete[] full_grad;
    delete[] x;
    delete[] y;
    if(is_store_result)
        return stored_F;
    return NULL;
}
