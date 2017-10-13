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

// Only for Ridge Regression
std::vector<double>* grad_desc_acc_dense::SVRG_LS(double* X, double* Y, size_t N, blackbox* model
    , size_t iteration_no, size_t interval, int Mode, double L, double step_size, double r, double* SV
    , bool is_store_result) {
    // Random Generator
    std::random_device rd;
    std::default_random_engine generator(rd());
    std::uniform_int_distribution<int> distribution(0, N - 1);
    std::vector<double>* stored_F = new std::vector<double>;
    double* inner_weights = new double[MAX_DIM];
    double* full_grad = new double[MAX_DIM];
    double* lambda = model->get_params();
    //FIXME: Epoch Size(SVRG / SVRG++)
    double m0 = (double) N * 2.0;
    size_t total_iterations = 0;
    copy_vec(inner_weights, model->get_model());
    // Init Weight Evaluate
    if(is_store_result)
        stored_F->push_back(model->zero_oracle_dense(X, Y, N));
    // OUTTER_LOOP
    for(size_t i = 0 ; i < iteration_no; i ++) {
        double* full_grad_core = new double[N];
        // Average Iterates
        double* aver_weights = new double[MAX_DIM];
        //FIXME: SVRG / SVRG++
        double inner_m = m0;//pow(2, i + 1) * m0;
        memset(aver_weights, 0, MAX_DIM * sizeof(double));
        memset(full_grad, 0, MAX_DIM * sizeof(double));
        // Full Gradient
        for(size_t j = 0; j < N; j ++) {
            full_grad_core[j] = model->first_component_oracle_core_dense(X, Y, N, j);
            for(size_t k = 0; k < MAX_DIM; k ++) {
                full_grad[k] += (X[j * MAX_DIM + k] * full_grad_core[j]) / (double) N;
            }
        }
        switch(Mode) {
            case SVRG_LAST_LAST:
            case SVRG_AVER_LAST:
                break;
            case SVRG_AVER_AVER:
                copy_vec(inner_weights, model->get_model());
                break;
            default:
                throw std::string("500 Internal Error.");
                break;
        }
        // INNER_LOOP
        for(size_t j = 0; j < inner_m ; j ++) {
            int rand_samp = distribution(generator);
            double inner_core = model->first_component_oracle_core_dense(X, Y, N
                , rand_samp, inner_weights);
            if(!((j + 1) % interval)) {
                double* vr_sub_grad = new double[MAX_DIM];
                memset(vr_sub_grad, 0, MAX_DIM * sizeof(double));
                double DAAX = 0.0, DAB = 0.0, DAAD = 0.0, XD = 0.0, DD = 0.0;
                for(size_t k = 0; k < N; k ++) {
                    double ADk = 0.0, AXk = 0.0;
                    for(size_t l = 0; l < MAX_DIM; l ++) {
                        if(k == 0) {
                            vr_sub_grad[l] = (inner_core - full_grad_core[rand_samp]) * X[rand_samp * MAX_DIM + l]
                                 + inner_weights[l]* lambda[0] + full_grad[l];
                            XD += inner_weights[l] * vr_sub_grad[l];
                            DD += vr_sub_grad[l] * vr_sub_grad[l];
                        }
                        double val = X[k * MAX_DIM + l];
                        ADk += val * vr_sub_grad[l];
                        AXk += val * inner_weights[l];
                    }
                    DAAX += ADk * AXk;
                    DAB += ADk * Y[k];
                    DAAD += ADk * ADk;
                }
                double alpha = (N * lambda[0] * XD + DAAX - DAB)
                        / (DAAD + N * lambda[0] * DD);
                for(size_t k = 0; k < MAX_DIM; k ++) {
                    inner_weights[k] -= alpha * vr_sub_grad[k];
                    aver_weights[k] += inner_weights[k] / inner_m;
                }
                delete vr_sub_grad;
            }
            else {
                for(size_t k = 0; k < MAX_DIM; k ++) {
                    double val = X[rand_samp * MAX_DIM + k];
                    double vr_sub_grad = (inner_core - full_grad_core[rand_samp]) * val
                             + inner_weights[k]* lambda[0] + full_grad[k];
                    inner_weights[k] -= step_size * vr_sub_grad;
                    aver_weights[k] += inner_weights[k] / inner_m;
                }
            }
            total_iterations ++;
        }
        switch(Mode) {
            case SVRG_LAST_LAST:
                model->update_model(inner_weights);
                break;
            case SVRG_AVER_LAST:
            case SVRG_AVER_AVER:
                model->update_model(aver_weights);
                break;
            default:
                throw std::string("500 Internal Error.");
                break;
        }
        // For Matlab (per m/n passes)
        if(is_store_result) {
            stored_F->push_back(model->zero_oracle_dense(X, Y, N));
        }
        delete[] aver_weights;
        delete[] full_grad_core;
    }
    delete[] full_grad;
    delete[] inner_weights;
    if(is_store_result)
        return stored_F;
    return NULL;
}
