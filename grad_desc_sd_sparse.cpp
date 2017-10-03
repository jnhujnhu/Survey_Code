#include "grad_desc_sd_sparse.hpp"
#include "utils.hpp"
#include "regularizer.hpp"
#include <random>
#include <cmath>
#include <complex>
#include <string.h>

extern size_t MAX_DIM;

std::vector<double>* grad_desc_sd_sparse::SVRG_SD(double* X, double* Y, size_t* Jc
    , size_t* Ir, size_t N, blackbox* model, size_t iteration_no, size_t interval
    , double L, double sigma, double step_size, bool is_store_weight, bool is_debug_mode
    , bool is_store_result) {
    // Random Generator
    std::random_device rd;
    std::default_random_engine generator(rd());
    std::uniform_int_distribution<int> distribution(0, N - 1);
    std::vector<double>* stored_F = new std::vector<double>;
    double* x = new double[MAX_DIM];
    double* y = new double[MAX_DIM];
    double* x_hat = new double[MAX_DIM];
    double* full_grad = new double[MAX_DIM];
    // Trade off parameter
    double delta = 0.1;
    double zeta = delta * step_size / (1.0 - L * step_size);
    double m0 = (double) N * 2.0;
    int regular = model->get_regularizer();
    double* lambda = model->get_params();
    size_t total_iterations = 0;
    // Compute (y^T)X
    double* yX = new double [MAX_DIM];
    memset(yX, 0, MAX_DIM * sizeof(double));
    for(size_t i = 0; i < N; i ++) {
        for(size_t j = Jc[i]; j < Jc[i + 1]; j ++) {
            yX[Ir[j]] += Y[i] * X[j];
        }
    }
    copy_vec(x, model->get_model());
    copy_vec(x_hat, model->get_model());
    copy_vec(y, model->get_model());
    // Init Weight Evaluate
    if(is_store_result)
        stored_F->push_back(model->zero_oracle_sparse(X, Y, Jc, Ir, N));
    // OUTTER_LOOP
    for(size_t i = 0 ; i < iteration_no; i ++) {
        double* full_grad_core = new double[N];
        // Average Iterates
        double* aver_weights = new double[MAX_DIM];
        double* prev_x_hat = new double[MAX_DIM];
        int* last_seen = new int[MAX_DIM];
        double inner_m = m0;
        memset(prev_x_hat, 0, MAX_DIM * sizeof(double));
        memset(aver_weights, 0, MAX_DIM * sizeof(double));
        memset(full_grad, 0, MAX_DIM * sizeof(double));
        for(size_t i = 0; i < MAX_DIM; i ++) last_seen[i] = -1;

        switch(regular) {
            case regularizer::L2:
            case regularizer::ELASTIC_NET:
                // FIXME: Update Options.
                // copy_vec(x, model->get_model());
                copy_vec(x_hat, x);
                break;
            case regularizer::L1:
                copy_vec(x, y);
                copy_vec(x_hat, y);
                break;
            default:
                throw std::string("Error2");
                break;
        }
        // Full Gradient
        for(size_t j = 0; j < N; j ++) {
            full_grad_core[j] = model->first_component_oracle_core_sparse(X, Y, Jc, Ir, N, j);
            for(size_t k = Jc[j]; k < Jc[j + 1]; k ++) {
                full_grad[Ir[k]] += X[k] * full_grad_core[j] / (double) N;
            }
        }
        // INNER_LOOP
        for(size_t j = 0; j < inner_m; j ++) {
            int rand_samp = distribution(generator);
            // Compute Theta_k (every 2000 iter)
            // Non-acc Normal Loop
            double theta = 1.0;
            // Acc-SD Loop
            if(!((j + 1) % interval)) {
                // Lazy Aggragate to Keep x Up-to-date.
                for(size_t k = 0; k < MAX_DIM; k ++) {
                    if((int)j > last_seen[k] + 1) {
                        aver_weights[k] += regularizer::proximal_operator(regular, x[k]
                            , step_size, lambda, j - (last_seen[k] + 1), true, -step_size * full_grad[k]) / inner_m;
                        last_seen[k] = j - 1;
                    }
                }
                double inner_core = model->first_component_oracle_core_sparse(X, Y, Jc, Ir, N
                                        , rand_samp, x);
                double* vr_sub_grad = new double[MAX_DIM];
                memset(vr_sub_grad, 0, MAX_DIM * sizeof(double));
                switch(regular) {
                    case regularizer::L2: {
                        double bAx = 0.0, square_p = 0.0, square_x = 0.0;
                        for(size_t k = 0; k < MAX_DIM; k ++) {
                            bAx += yX[k] * x[k];
                            square_x += x[k] * x[k];
                        }
                        for(size_t k = Jc[rand_samp]; k < Jc[rand_samp + 1]; k ++) {
                            double difference = (inner_core - full_grad_core[rand_samp])
                                    * X[k];
                            vr_sub_grad[Ir[k]] = difference;
                            square_p += difference * difference;
                        }
                        bAx /= (double) N;
                        double Ax = 0.0;
                        for(size_t k = 0; k < N; k ++) {
                            double Ad = 0.0;
                            for(size_t l = Jc[k]; l < Jc[k + 1]; l ++)
                                Ad += X[l] * x[Ir[l]];
                            Ax += Ad * Ad;
                        }
                        theta = (bAx + zeta * square_p)
                            / (Ax / N + zeta * square_p + lambda[0] * square_x);
                        break;
                    }
                    case regularizer::L1: {
                        throw std::string("Not Done.");
                        break;
                    }
                    default:
                        throw std::string("Error");
                        break;
                }
                // Acc-SD Update x
                for(size_t k = 0; k < MAX_DIM; k ++) {
                    y[k] = x[k] - step_size * (vr_sub_grad[k] + full_grad[k]);
                    regularizer::proximal_operator(regular, y[k], step_size, lambda);
                    prev_x_hat[k] = x_hat[k];
                    x_hat[k] = theta * x[k];
                    x[k] = y[k] + (1.0 - sigma) * (x_hat[k] - prev_x_hat[k]);
                    aver_weights[k] += x_hat[k] / (double) inner_m;
                    last_seen[k] = j;
                }
            }
            else {
                for(size_t k = Jc[rand_samp]; k < Jc[rand_samp + 1]; k ++) {
                    size_t index = Ir[k];
                    if((int)j > last_seen[index] + 1) {
                        // Lazy Update
                        aver_weights[index] += regularizer::proximal_operator(regular, x[index]
                            , step_size, lambda, j - (last_seen[index] + 1), true, -step_size * full_grad[index]) / inner_m;
                    }
                }
                double inner_core = model->first_component_oracle_core_sparse(X, Y, Jc, Ir, N
                                        , rand_samp, x);
                for(size_t k = Jc[rand_samp]; k < Jc[rand_samp + 1]; k ++) {
                    size_t index = Ir[k];
                    double val = X[k];
                    double vr_sub_grad = (inner_core - full_grad_core[rand_samp]) * val + full_grad[index];
                    // Non Acc Normal Update x
                    x[index] -= step_size * vr_sub_grad;
                    regularizer::proximal_operator(regular, x[index], step_size, lambda);
                    aver_weights[index] += x[index] / (double) inner_m;
                    last_seen[index] = j;
                }
            }
            total_iterations ++;
        }
        for(size_t j = 0; j < MAX_DIM; j ++) {
            if(inner_m > last_seen[j] + 1) {
                // Lazy Aggragate
                aver_weights[j] += regularizer::proximal_operator(regular, x[j]
                    , step_size, lambda, inner_m - (last_seen[j] + 1), true, -step_size * full_grad[j]) / inner_m;
            }
        }
        model->update_model(aver_weights);
        // NSC update y
        if(regular == regularizer::L1) {
            for(size_t k = 0; k < MAX_DIM; k ++)
                y[k] =  (x[k] - (1 - sigma) * x_hat[k]) / sigma;
        }
        // For Matlab (per m/n passes)
        if(is_store_result) {
            stored_F->push_back(model->zero_oracle_sparse(X, Y, Jc, Ir, N));
        }
        delete[] aver_weights;
        delete[] full_grad_core;
        delete[] prev_x_hat;
        delete[] last_seen;
    }
    delete[] full_grad;
    delete[] x;
    delete[] y;
    delete[] x_hat;
    if(is_store_result)
        return stored_F;
    return NULL;
}

std::vector<double>* grad_desc_sd_sparse::SAGA_SD(double* X, double* Y, size_t* Jc, size_t* Ir, size_t N, blackbox* model
    , size_t iteration_no, double L, double step_size, double r, double* SV, bool is_store_weight
    , bool is_debug_mode, bool is_store_result) {
    return NULL;
}
