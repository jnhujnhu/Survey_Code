#include "grad_desc_MiG.hpp"
#include "utils.hpp"
#include "regularizer.hpp"
#include <random>
#include <cmath>
#include <string.h>

extern size_t MAX_DIM;

std::vector<double>* grad_desc_MiG::Ladder_SVRG(double* X, double* Y, size_t N, blackbox* model
    , size_t iteration_no, int Mode, double L, double sigma, double step_size
    , bool is_store_result) {
    // Random Generator
    std::random_device rd;
    std::default_random_engine generator(rd());
    std::uniform_int_distribution<int> distribution(0, N - 1);
    std::uniform_real_distribution<> magic(0.0, 1.0);
    // std::vector<double>* stored_weights = new std::vector<double>;
    std::vector<double>* stored_F = new std::vector<double>;
    double* inner_weights = new double[MAX_DIM];
    double* full_grad = new double[MAX_DIM];
    //FIXME: Epoch Size(SVRG / SVRG++)
    double m0 = (double) N * 2.0;
    int regular = model->get_regularizer();
    double* lambda = model->get_params();
    size_t total_iterations = 0;
    copy_vec(inner_weights, model->get_model());
    // Init Weight Evaluate
    if(is_store_result)
        stored_F->push_back(model->zero_oracle_dense(X, Y, N));
    double* full_grad_core = new double[N];
    // OUTTER_LOOP
    for(size_t i = 0 ;  i < iteration_no; i ++) {
        // Average Iterates
        double* aver_weights = new double[MAX_DIM];
        //FIXME: SVRG / SVRG++
        double inner_m = m0;//pow(2, i + 1) * m0;
        memset(aver_weights, 0, MAX_DIM * sizeof(double));
        // TEST
        double flag = magic(generator);
        if(i == 0 || flag >= sigma) {
            memset(full_grad, 0, MAX_DIM * sizeof(double));
            // Full Gradient
            for(size_t j = 0; j < N; j ++) {
                full_grad_core[j] = model->first_component_oracle_core_dense(X, Y, N, j);
                for(size_t k = 0; k < MAX_DIM; k ++) {
                    full_grad[k] += X[j * MAX_DIM + k] * full_grad_core[j] / (double) N;
                }
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
                throw std::string("400 Unrecognized Mode.");
                break;
        }
        // INNER_LOOP
        for(size_t j = 0; j < inner_m; j ++) {
            int rand_samp = distribution(generator);
            double inner_core = model->first_component_oracle_core_dense(X, Y, N
                , rand_samp, inner_weights);
            for(size_t k = 0; k < MAX_DIM; k ++) {
                double val = X[rand_samp * MAX_DIM + k];
                double vr_sub_grad = (inner_core - full_grad_core[rand_samp]) * val + full_grad[k];
                inner_weights[k] -= step_size * vr_sub_grad;
                aver_weights[k] += regularizer::proximal_operator(regular, inner_weights[k], step_size, lambda) / inner_m;
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
    }
    delete[] full_grad_core;
    delete[] full_grad;
    delete[] inner_weights;
    if(is_store_result)
        return stored_F;
    return NULL;
}

std::vector<double>* grad_desc_MiG::Ladder_SVRG_sparse(double* X, double* Y, size_t* Jc, size_t* Ir
    , size_t N, blackbox* model, size_t iteration_no, int Mode, double L, double sigma, double step_size
    , bool is_store_result) {
    // Random Generator
    std::random_device rd;
    std::default_random_engine generator(rd());
    std::uniform_int_distribution<int> distribution(0, N - 1);
    std::uniform_real_distribution<> magic(0.0, 1.0);
    std::vector<double>* stored_F = new std::vector<double>;
    double* inner_weights = new double[MAX_DIM];
    double* full_grad = new double[MAX_DIM];
    //FIXME: Epoch Size(SVRG / SVRG++)
    double m0 = (double) N * 2.0;
    int regular = model->get_regularizer();
    double* lambda = model->get_params();
    size_t total_iterations = 0;
    copy_vec(inner_weights, model->get_model());
    // Init Weight Evaluate
    if(is_store_result)
        stored_F->push_back(model->zero_oracle_sparse(X, Y, Jc, Ir, N));
    double* full_grad_core = new double[N];
    // OUTTER_LOOP
    for(size_t i = 0 ; i < iteration_no; i ++) {
        // Average Iterates
        double* aver_weights = new double[MAX_DIM];
        // lazy update extra params.
        int* last_seen = new int[MAX_DIM];
        //FIXME: SVRG / SVRG++
        double inner_m = m0;//pow(2, i + 1) * m0;
        memset(aver_weights, 0, MAX_DIM * sizeof(double));
        for(size_t i = 0; i < MAX_DIM; i ++) last_seen[i] = -1;
        double flag = magic(generator);
        if(i == 0 || flag >= sigma) {
            memset(full_grad, 0, MAX_DIM * sizeof(double));
            // Full Gradient
            for(size_t j = 0; j < N; j ++) {
                full_grad_core[j] = model->first_component_oracle_core_sparse(X, Y, Jc, Ir, N, j);
                for(size_t k = Jc[j]; k < Jc[j + 1]; k ++) {
                    full_grad[Ir[k]] += X[k] * full_grad_core[j] / (double) N;
                }
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
        for(size_t j = 0; j < inner_m; j ++) {
            int rand_samp = distribution(generator);
            for(size_t k = Jc[rand_samp]; k < Jc[rand_samp + 1]; k ++) {
                size_t index = Ir[k];
                // lazy update
                if((int)j > last_seen[index] + 1) {
                    switch(Mode) {
                        case SVRG_LAST_LAST:
                            regularizer::proximal_operator(regular, inner_weights[index], step_size
                                , lambda, j - (last_seen[index] + 1), false, -step_size * full_grad[index]);
                            break;
                        case SVRG_AVER_LAST:
                        case SVRG_AVER_AVER:
                            aver_weights[index] += regularizer::proximal_operator(regular, inner_weights[index]
                                , step_size, lambda, j - (last_seen[index] + 1), true, -step_size * full_grad[index]) / inner_m;
                            break;
                        default:
                            throw std::string("500 Internal Error.");
                            break;
                    }
                }
            }
            double inner_core = model->first_component_oracle_core_sparse(X, Y
                        , Jc, Ir, N, rand_samp, inner_weights);
            for(size_t k = Jc[rand_samp]; k < Jc[rand_samp + 1]; k ++) {
                size_t index = Ir[k];
                double val = X[k];
                double vr_sub_grad = (inner_core - full_grad_core[rand_samp]) * val + full_grad[index];
                inner_weights[index] -= step_size * vr_sub_grad;
                aver_weights[index] += regularizer::proximal_operator(regular, inner_weights[index]
                    , step_size, lambda) / inner_m;
                last_seen[index] = j;
            }
            total_iterations ++;
        }
        // lazy update aggragate
        switch(Mode) {
            case SVRG_LAST_LAST:
                for(size_t j = 0; j < MAX_DIM; j ++) {
                    if(inner_m > last_seen[j] + 1) {
                        regularizer::proximal_operator(regular, inner_weights[j], step_size, lambda
                            , inner_m - (last_seen[j] + 1), false, -step_size * full_grad[j]);
                    }
                }
                model->update_model(inner_weights);
                break;
            case SVRG_AVER_LAST:
            case SVRG_AVER_AVER:
                for(size_t j = 0; j < MAX_DIM; j ++) {
                    if(inner_m > last_seen[j] + 1) {
                        aver_weights[j] += regularizer::proximal_operator(regular, inner_weights[j], step_size
                            , lambda, inner_m - (last_seen[j] + 1), true, -step_size * full_grad[j]) / inner_m;
                    }
                }
                model->update_model(aver_weights);
                break;
            default:
                throw std::string("500 Internal Error.");
                break;
        }
        // For Matlab (per m/n passes)
        if(is_store_result) {
            stored_F->push_back(model->zero_oracle_sparse(X, Y, Jc, Ir, N));
        }
        delete[] last_seen;
        delete[] aver_weights;
    }
    delete[] full_grad_core;
    delete[] full_grad;
    delete[] inner_weights;
    if(is_store_result)
        return stored_F;
    return NULL;
}
