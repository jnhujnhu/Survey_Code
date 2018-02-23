#include "grad_desc_MiG.hpp"
#include "utils.hpp"
#include "regularizer.hpp"
#include <random>
#include <cmath>
#include <string.h>
#include <iostream>

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

std::vector<double>* grad_desc_MiG::SARAH(double* X, double* Y, size_t N, blackbox* model
    , size_t iteration_no, int Mode, double L, double sigma, double step_size
    , bool is_store_result) {
    // Random Generator
    std::random_device rd;
    std::default_random_engine generator(rd());
    std::uniform_int_distribution<int> distribution(0, N - 1);
    // std::vector<double>* stored_weights = new std::vector<double>;
    std::vector<double>* stored_F = new std::vector<double>;
    double* x = new double[MAX_DIM];
    //FIXME: Epoch Size(SVRG / SVRG++)
    double m0 = (double) N * 2.8;
    int regular = model->get_regularizer();
    double* lambda = model->get_params();
    size_t total_iterations = 0;
    copy_vec(x, model->get_model());
    // Init Weight Evaluate
    if(is_store_result)
        stored_F->push_back(model->zero_oracle_dense(X, Y, N));
    // OUTTER_LOOP
    while(total_iterations <= iteration_no) {
        // Average Iterates
        double* aver_weights = new double[MAX_DIM];
        double* prev_x = new double[MAX_DIM];
        double* v = new double[MAX_DIM];
        //FIXME: SVRG / SVRG++
        double inner_m = m0;//pow(2, i + 1) * m0;
        memset(aver_weights, 0, MAX_DIM * sizeof(double));
        memset(v, 0, MAX_DIM * sizeof(double));
        // Full Gradient
        for(size_t j = 0; j < N; j ++) {
            double core = model->first_component_oracle_core_dense(X, Y, N, j);
            for(size_t k = 0; k < MAX_DIM; k ++) {
                v[k] += X[j * MAX_DIM + k] * core/ (double) N;
            }
            total_iterations ++;
            // For Matlab
            if(!(total_iterations % (3 * N)) && is_store_result) {
                stored_F->push_back(model->zero_oracle_dense(X, Y, N));
            }
        }
        double v_0 = comp_l2_norm(v);
        copy_vec(prev_x, x);
        for(size_t j = 0; j < MAX_DIM; j ++)
            x[j] -= step_size * v[j];
        // INNER_LOOP
        double gamma = sqrt(1.0 / 64.0);
        size_t iter = 0;
        while(comp_l2_norm(v) > gamma * v_0 && iter < inner_m) {
            iter ++;
            int rand_samp = distribution(generator);
            double inner_core = model->first_component_oracle_core_dense(X, Y, N
                , rand_samp, x);
            double prev_core = model->first_component_oracle_core_dense(X, Y, N
                , rand_samp, prev_x);
            for(size_t k = 0; k < MAX_DIM; k ++) {
                double val = X[rand_samp * MAX_DIM + k];
                v[k] += (inner_core - prev_core) * val;
                prev_x[k] = x[k];
                x[k] -= step_size * v[k];
                aver_weights[k] += regularizer::proximal_operator(regular, x[k], step_size, lambda) / inner_m;
            }
            total_iterations ++;
            model->update_model(x);
            // For Matlab
            if(!(total_iterations % (3 * N)) && is_store_result) {
                stored_F->push_back(model->zero_oracle_dense(X, Y, N));
            }
        }
        std::cout << iter << std::endl;
        delete[] v;
        delete[] prev_x;
        delete[] aver_weights;
    }
    delete[] x;
    if(is_store_result)
        return stored_F;
    return NULL;
}

std::vector<double>* grad_desc_MiG::SARAH2(double* X, double* Y, size_t N, blackbox* model
    , size_t iteration_no, int Mode, double L, double sigma, double step_size
    , bool is_store_result) {
    // Random Generator
    std::random_device rd;
    std::default_random_engine generator(rd());
    std::uniform_int_distribution<int> distribution(0, N - 1);
    // std::vector<double>* stored_weights = new std::vector<double>;
    std::vector<double>* stored_F = new std::vector<double>;
    double* x = new double[MAX_DIM];
    //FIXME: Epoch Size(SVRG / SVRG++)
    double m0 = (double) N * 2.0;
    size_t total_iterations = 0;
    copy_vec(x, model->get_model());
    // Init Weight Evaluate
    if(is_store_result)
        stored_F->push_back(model->zero_oracle_dense(X, Y, N));
    double* aver_x = new double[MAX_DIM];
    double* prev_x = new double[MAX_DIM];
    double* v = new double[MAX_DIM];
    double* record_x = new double[MAX_DIM];
    double* aver_inn = new double[MAX_DIM];
    double* grad_table = new double[N];
    // OUTTER_LOOP
    for(size_t i = 0; i < iteration_no; i ++) {
        //FIXME: SVRG / SVRG++
        double inner_m = m0;//pow(2, i + 1) * m0;
        if(i < 5) {
            memset(aver_x, 0, MAX_DIM * sizeof(double));
            memset(v, 0, MAX_DIM * sizeof(double));
            // Full Gradient
            for(size_t j = 0; j < N; j ++) {
                double core = model->first_component_oracle_core_dense(X, Y, N, j);
                grad_table[j] = core;
                for(size_t k = 0; k < MAX_DIM; k ++) {
                    aver_x[k] += X[j * MAX_DIM + k] * core/ (double) N;
                }
            }
            copy_vec(prev_x, x);
            for(size_t j = 0; j < MAX_DIM; j ++)
                x[j] -= step_size * aver_x[j];
        }
        else {
            //copy_vec(aver_x, aver_inn);
            //memset(v, 0, MAX_DIM * sizeof(double));
        }
        memset(aver_inn, 0, MAX_DIM * sizeof(double));
        // INNER_LOOP
        for(size_t j = 0; j < inner_m; j ++) {
            int rand_samp = distribution(generator);
            double inner_core = model->first_component_oracle_core_dense(X, Y, N
                , rand_samp, x);
            double prev_core = model->first_component_oracle_core_dense(X, Y, N
                , rand_samp, prev_x);
            double past_core = grad_table[rand_samp];
            grad_table[rand_samp] = inner_core;
            for(size_t k = 0; k < MAX_DIM; k ++) {
                double val = X[rand_samp * MAX_DIM + k];
                double bias = 0;
                if(i != 0) {
                    v[k] -=  prev_core * val;
                    //aver_x[k] += (inner_core - past_core) * val / N;
                    // bias = (inner_core * val - record_x[k]) / N + (re);
                    // record_x[k] = inner_core * val;
                }
                else if(j == 0) {
                    record_x[k] = -prev_core * val;
                    bias = record_x[k];
                }
                else {
                    v[k] -=  prev_core * val;
                    //aver_x[k] += (inner_core - past_core) * val / N;
                    // bias = (inner_core * val - record_x[k]) / N;
                    // record_x[k] = inner_core * val;
                }
                prev_x[k] = x[k];
                double inn_grad = (inner_core * val + v[k] + aver_x[k] + bias);
                x[k] -= step_size * inn_grad;
                aver_inn[k] += inn_grad / inner_m;
                v[k] = v[k] + (inner_core) * val;
            }
            total_iterations ++;
        }
        model->update_model(x);
        // For Matlab
        if(is_store_result) {
            stored_F->push_back(model->zero_oracle_dense(X, Y, N));
        }
    }
    delete[] grad_table;
    delete[] aver_inn;
    delete[] v;
    delete[] prev_x;
    delete[] aver_x;
    delete[] record_x;
    delete[] x;
    if(is_store_result)
        return stored_F;
    return NULL;
}


double logistic_hessian_vector_oracle_core(double* X, double* Y, size_t given_index,
    double* x, double* v) {
    double sigmoid = 0.0, innr_Xv = 0.0;
    for(size_t j = 0; j < MAX_DIM; j ++) {
        double Xj = X[given_index * MAX_DIM + j];
        sigmoid += x[j] * Xj;
        innr_Xv += v[j] * Xj;
    }
    sigmoid = exp(sigmoid * - Y[given_index]);
    sigmoid = sigmoid /  (1 + sigmoid);
    return sigmoid * (1 - sigmoid) * Y[given_index] * Y[given_index] * innr_Xv;
}

// TEST
std::vector<double>* grad_desc_MiG::LSAGA(double* X, double* Y, size_t N, blackbox* model
    , size_t iteration_no, double L, double step_size, bool is_store_result) {
    // Random Generator
    std::random_device rd;
    std::default_random_engine generator(rd());
    std::uniform_int_distribution<int> distribution(0, N - 1);
    std::vector<double>* stored_F = new std::vector<double>;
    // For Matlab
    if(is_store_result) {
        stored_F->push_back(model->zero_oracle_dense(X, Y, N));
    }
    double* w = new double[MAX_DIM];
    double* v_w = new double[MAX_DIM];
    double* aver_w = new double[MAX_DIM];
    double* prev_w = new double[MAX_DIM];
    int* past_samples = new int[N];
    memset(aver_w, 0, MAX_DIM * sizeof(double));
    memset(v_w, 0, MAX_DIM * sizeof(double));
    // Start Up
    double* table = new double[N];
    for(size_t j = 0; j < N; j ++) {
        double core = model->first_component_oracle_core_dense(X, Y, N, j);
        table[j] = core;
        for(size_t k = 0; k < MAX_DIM; k ++) {
            aver_w[k] += X[j * MAX_DIM + k] * core/ (double) N;
        }
    }
    copy_vec(w, model->get_model());
    //copy_vec(prev_w, w);
    for(size_t j = 0; j < iteration_no; j ++) {
        int rand_samp = distribution(generator);
        double inner_core = model->first_component_oracle_core_dense(X, Y, N
            , rand_samp, w);

        // double prev_core = model->first_component_oracle_core_dense(X, Y, N
        //     , rand_samp, prev_w);
        // if(j >= N) {
        //     double past_core = model->first_component_oracle_core_dense(X, Y, N
        //         , past_samples[j % N], prev_w);
        //     for(size_t k = 0; k < MAX_DIM; k ++) {
        //         st += past_core *
        //     }
        // }
        // past_samples[j % N] = rand_samp;
        for(size_t k = 0; k < MAX_DIM; k ++) {
            double val = X[rand_samp * MAX_DIM + k];
            // double bias = 0;
            // if(j == 0) {
            //     bias = -prev_core * val;
            // }
            // else
            //     v_w[k] -= prev_core * val;
            // prev_w[k] = w[k];
            aver_w[k] += (inner_core - table[j % N]) * val / N;
            w[k] -= step_size * (aver_w[k]);
            //v_w[k] = v_w[k] + inner_core * val;
        }
        table[j % N] = inner_core;
        if(!(j % (3 * N))) {
            model->update_model(w);
            stored_F->push_back(model->zero_oracle_dense(X, Y, N));
        }
    }
    model->update_model(w);
    delete[] table;
    delete[] w;
    delete[] v_w;
    delete[] aver_w;
    delete[] prev_w;
    delete[] past_samples;
    // For Matlab
    if(is_store_result)
        return stored_F;
    return NULL;
}
