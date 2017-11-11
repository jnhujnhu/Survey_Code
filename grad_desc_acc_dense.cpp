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
        double* aver_weights = new double[MAX_DIM];
        memset(aver_weights, 0, MAX_DIM * sizeof(double));
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
                aver_weights[k] += x[k] / inner_m;
            }
            total_iterations ++;
        }
        model->update_model(aver_weights);
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

/////////CONTAINS BUG, DO NOT CONVERGE/////////////
std::vector<double>* grad_desc_acc_dense::FSVRG(double* X, double* Y, size_t N, blackbox* model
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
    double m0 = (double) N;
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
        double* aver_weights = new double[MAX_DIM];
        memset(aver_weights, 0, MAX_DIM * sizeof(double));
        double inner_m = m0 * 2;
        double theta = sigma * step_size * inner_m / 2.0;
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
                , rand_samp, x);
            for(size_t k = 0; k < MAX_DIM; k ++) {
                double val = X[rand_samp * MAX_DIM + k];
                double vr_sub_grad = (inner_core - full_grad_core[rand_samp]) * val + full_grad[k];
                y[k] -= step_size / theta * vr_sub_grad;
                y[k] = regularizer::proximal_operator(regular, y[k], step_size, lambda);
                x[k] = y[k] + (1.0 - theta) * (model->get_model()[k] - y[k]);
                aver_weights[k] += x[k] / inner_m;
            }
            total_iterations ++;
        }
        model->update_model(aver_weights);
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
    , size_t iteration_no, size_t interval, int Mode, int LSF_Mode, int LSC_Mode, int LSM_Mode, double L
    , double step_size, double r, double* SV, bool is_store_result) {
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

    double* ATA = new double[MAX_DIM * MAX_DIM];
    double* ATb = new double[MAX_DIM];

    // Compute ATA;
    for(size_t i = 0; i < MAX_DIM; i ++) {
        for(size_t j = 0; j < MAX_DIM; j ++) {
            double temp = 0;
            for(size_t k = 0; k < N; k ++) {
                temp += X[k * MAX_DIM + i] * X[k* MAX_DIM + j];
            }
            ATA[i * MAX_DIM + j] = temp;
        }
    }

    // Compute ATb;
    for(size_t i = 0; i < MAX_DIM; i ++) {
        double temp = 0;
        for(size_t j = 0; j < N; j ++) {
            temp += X[j * MAX_DIM + i] * Y[j];
        }
        ATb[i] = temp;
    }

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
        double* full_LSC_core = NULL;
        switch(LSC_Mode) {
            case SVRG_LS_OUTF:
                break;
            case SVRG_LS_CHGF:
                full_LSC_core = new double[N];
                memset(full_LSC_core, 0, N * sizeof(double));
                break;
            default:
                throw std::string("500 Internal Error.");
                break;
        }
        double* ls_grad = new double[MAX_DIM];
        bool ls_flag = false;
        // INNER_LOOP
        for(size_t j = 0; j < inner_m ; j ++) {
            int rand_samp = distribution(generator);
            double inner_core = model->first_component_oracle_core_dense(X, Y, N
                , rand_samp, inner_weights);
            if(!((j + 1) % interval)) {
                memset(ls_grad, 0, MAX_DIM * sizeof(double));
                double DAAX = 0.0, DAB = 0.0, DAAD = 0.0, XD = 0.0, DD = 0.0;
                switch (LSF_Mode) {
                    case SVRG_LS_FULL:{
                        // Full Gradient
                        for(size_t j = 0; j < N; j ++) {
                            switch(LSC_Mode) {
                                case SVRG_LS_OUTF:{
                                    double temp_full_core = model->first_component_oracle_core_dense(X, Y, N, j, inner_weights);
                                    for(size_t k = 0; k < MAX_DIM; k ++)
                                        ls_grad[k] += (X[j * MAX_DIM + k] * temp_full_core) / (double) N;
                                    break;
                                }
                                case SVRG_LS_CHGF:
                                    ls_flag = true;
                                    full_LSC_core[j] = model->first_component_oracle_core_dense(X, Y, N, j, inner_weights);
                                    for(size_t k = 0; k < MAX_DIM; k ++)
                                        ls_grad[k] += (X[j * MAX_DIM + k] * full_LSC_core[j]) / (double) N;
                                    break;
                            }
                        }
                        for(size_t k = 0; k < MAX_DIM; k ++) {
                            XD += inner_weights[k] * ls_grad[k];
                            DD += ls_grad[k] * ls_grad[k];
                        }
                        break;
                    }
                    case SVRG_LS_STOC:{
                        for(size_t k = 0; k < MAX_DIM; k ++) {
                            ls_grad[k] = (inner_core - full_grad_core[rand_samp]) * X[rand_samp * MAX_DIM + k]
                                 + inner_weights[k]* lambda[0] + full_grad[k];
                            XD += inner_weights[k] * ls_grad[k];
                            DD += ls_grad[k] * ls_grad[k];
                        }
                        break;
                    }
                    default:
                        throw std::string("500 Internal Error.");
                        break;
                }
                switch (LSM_Mode) {
                    case SVRG_LS_A:
                        for(size_t k = 0; k < MAX_DIM; k ++) {
                            double temp = 0;
                            for(size_t l = 0; l < MAX_DIM; l ++) {
                                temp += ls_grad[l] * ATA[l * MAX_DIM + k];
                            }
                            DAAX += temp * inner_weights[k];
                            DAB += ls_grad[k] * ATb[k];
                        }
                        for(size_t k = 0; k < N; k ++) {
                            double ADk = 0.0;
                            for(size_t l = 0; l < MAX_DIM; l ++) {
                                ADk += X[k * MAX_DIM + l] * ls_grad[l];
                            }
                            DAAD += ADk * ADk;
                        }
                        break;
                    case SVRG_LS_SVD:
                        for(size_t k = 0; k < MAX_DIM; k ++) {
                            double temp = 0;
                            for(size_t l = 0; l < MAX_DIM; l ++) {
                                temp += ls_grad[l] * ATA[l * MAX_DIM + k];
                            }
                            DAAX += temp * inner_weights[k];
                            DAB += ls_grad[k] * ATb[k];
                        }
                        for(size_t k = 0; k < r; k ++) {
                            double ADk = 0.0;
                            for(size_t l = 0; l < MAX_DIM; l ++) {
                                ADk += SV[k * MAX_DIM + l] * ls_grad[l];
                            }
                            DAAD += ADk * ADk;
                        }
                        break;
                }
                double alpha = (N * lambda[0] * XD + DAAX - DAB)
                        / (DAAD + N * lambda[0] * DD);
                for(size_t k = 0; k < MAX_DIM; k ++) {
                    inner_weights[k] -= alpha * ls_grad[k];
                    aver_weights[k] += inner_weights[k] / inner_m;
                }
            }
            else {
                for(size_t k = 0; k < MAX_DIM; k ++) {
                    double val = X[rand_samp * MAX_DIM + k];
                    double vr_sub_grad = 0.0;
                    switch(LSC_Mode) {
                        case SVRG_LS_OUTF:
                            vr_sub_grad = (inner_core - full_grad_core[rand_samp]) * val
                                 + inner_weights[k]* lambda[0] + full_grad[k];
                            break;
                        case SVRG_LS_CHGF:
                            if(ls_flag)
                                vr_sub_grad = (inner_core - full_LSC_core[rand_samp]) * val
                                     + inner_weights[k]* lambda[0] + ls_grad[k];
                            else
                                vr_sub_grad = (inner_core - full_grad_core[rand_samp]) * val
                                     + inner_weights[k]* lambda[0] + full_grad[k];
                            break;
                    }
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
        delete[] full_LSC_core;
        delete[] ls_grad;
    }
    delete[] full_grad;
    delete[] inner_weights;
    if(is_store_result)
        return stored_F;
    return NULL;
}

std::vector<double>* grad_desc_acc_dense::Prox_SVRG_CP(double* X, double* Y, size_t N, blackbox* model
    , size_t iteration_no, int Mode, double L, double step_size, bool is_store_result) {
    // Random Generator
    std::random_device rd;
    std::default_random_engine generator(rd());
    std::uniform_int_distribution<int> distribution(0, N - 1);
    // std::vector<double>* stored_weights = new std::vector<double>;
    std::vector<double>* stored_F = new std::vector<double>;
    double* inner_weights = new double[MAX_DIM];
    double* full_grad = new double[MAX_DIM];
    double m0 = (double) N * 2.0;
    int regular = model->get_regularizer();
    double* lambda = model->get_params();
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
        double inner_m = m0;
        memset(aver_weights, 0, MAX_DIM * sizeof(double));
        memset(full_grad, 0, MAX_DIM * sizeof(double));
        // Full Gradient
        for(size_t j = 0; j < N; j ++) {
            full_grad_core[j] = model->first_component_oracle_core_dense(X, Y, N, j);
            for(size_t k = 0; k < MAX_DIM; k ++) {
                full_grad[k] += X[j * MAX_DIM + k] * full_grad_core[j] / (double) N;
            }
        }
        // Compute Hyperplane Coeff
        double hyper_c = 0.0, inner_F = 0.0;
        double* outter_x = model->get_model();
        double* prev_x = new double[MAX_DIM];
        double* _pF = new double[MAX_DIM];
        double* _pR = new double[MAX_DIM];
        regularizer::first_oracle(regular, _pR, lambda, outter_x);
        for(size_t j = 0; j < MAX_DIM; j ++) {
            double _F = full_grad[j] + _pR[j];
            hyper_c += outter_x[j] * _F;
            inner_F += _F * _F;
            _pF[j] = _F;
        }
        delete[] _pR;
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
            double hyper_x = 0.0;
            for(size_t k = 0; k < MAX_DIM; k ++) {
                double val = X[rand_samp * MAX_DIM + k];
                double vr_sub_grad = (inner_core - full_grad_core[rand_samp]) * val + full_grad[k];
                inner_weights[k] -= step_size * vr_sub_grad;
                aver_weights[k] += regularizer::proximal_operator(regular, inner_weights[k], step_size, lambda) / inner_m;
                prev_x[k] = inner_weights[k];
                hyper_x += inner_weights[k] * _pF[k];
            }
            // Away from Hyperplane
            if(hyper_x > hyper_c) {
                for(size_t k = 0; k < MAX_DIM; k ++) {
                    inner_weights[k] = inner_weights[k] - _pF[k]
                                    * (hyper_x / inner_F - hyper_c / inner_F);
                    aver_weights[k] += (inner_weights[k] - prev_x[k]) / inner_m;
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
        delete[] prev_x;
        delete[] _pF;
    }
    delete[] full_grad;
    delete[] inner_weights;
    if(is_store_result)
        return stored_F;
    return NULL;
}

std::vector<double>* grad_desc_acc_dense::Prox_SVRG_SCP(double* X, double* Y, size_t N, blackbox* model
    , size_t iteration_no, int Mode, double L, double step_size, bool is_store_result) {
    // Random Generator
    std::random_device rd;
    std::default_random_engine generator(rd());
    std::uniform_int_distribution<int> distribution(0, N - 1);
    // std::vector<double>* stored_weights = new std::vector<double>;
    std::vector<double>* stored_F = new std::vector<double>;
    double* inner_weights = new double[MAX_DIM];
    double* full_grad = new double[MAX_DIM];
    double m0 = (double) N * 2.0;
    int regular = model->get_regularizer();
    double* lambda = model->get_params();
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
        double inner_m = m0;
        memset(aver_weights, 0, MAX_DIM * sizeof(double));
        memset(full_grad, 0, MAX_DIM * sizeof(double));
        // Full Gradient
        for(size_t j = 0; j < N; j ++) {
            full_grad_core[j] = model->first_component_oracle_core_dense(X, Y, N, j);
            for(size_t k = 0; k < MAX_DIM; k ++) {
                full_grad[k] += X[j * MAX_DIM + k] * full_grad_core[j] / (double) N;
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
        // Stochastic Hyperplane Coeffs
        double* grad_core_table = new double[N * MAX_DIM];
        double* hyper_c_table = new double[N];
        memset(grad_core_table, 0, N * MAX_DIM * sizeof(double));
        memset(hyper_c_table, 0, N * sizeof(double));
        // INNER_LOOP
        for(size_t j = 0; j < inner_m; j ++) {
            int rand_samp = distribution(generator);
            double inner_core = model->first_component_oracle_core_dense(X, Y, N
                , rand_samp, inner_weights);
            double* prev_x = new double[MAX_DIM];
            double* prev_x2 = new double[MAX_DIM];
            double hyper_x = 0;
            for(size_t k = 0; k < MAX_DIM; k ++) {
                double val = X[rand_samp * MAX_DIM + k];
                double vr_sub_grad = (inner_core - full_grad_core[rand_samp]) * val + full_grad[k];
                prev_x[k] = inner_weights[k];
                inner_weights[k] -= step_size * vr_sub_grad;
                aver_weights[k] += regularizer::proximal_operator(regular, inner_weights[k], step_size, lambda) / inner_m;
                prev_x2[k] = inner_weights[k];
                if(hyper_c_table[rand_samp] != 0)
                    hyper_x += inner_weights[k] * grad_core_table[rand_samp * MAX_DIM + k];
            }
            if(hyper_x < hyper_c_table[rand_samp] || hyper_c_table[rand_samp] == 0){
                double hyper_c = 0;
                double* _pR = new double[MAX_DIM];
                regularizer::first_oracle(regular, _pR, lambda, prev_x);
                for(size_t k = 0; k < MAX_DIM; k ++) {
                    double val = X[rand_samp * MAX_DIM + k];
                    double vr_sub_grad = (inner_core - full_grad_core[rand_samp]) * val
                                + full_grad[k] + _pR[k];
                    hyper_c += prev_x[k] * vr_sub_grad;
                    grad_core_table[rand_samp * MAX_DIM + k] = vr_sub_grad;
                }
                hyper_c_table[rand_samp] = hyper_c;
                delete[] _pR;
            }
            // Away from stochastic hyperplane
            else if(hyper_x > hyper_c_table[rand_samp]) {
                double inner_F = 0;
                for(size_t k = 0; k < MAX_DIM; k ++) {
                    inner_F += grad_core_table[rand_samp * MAX_DIM + k]
                            * grad_core_table[rand_samp * MAX_DIM + k];
                }
                for(size_t k = 0; k < MAX_DIM; k ++) {
                    inner_weights[k] = inner_weights[k] - grad_core_table[rand_samp * MAX_DIM + k]
                                * (hyper_x / inner_F - hyper_c_table[rand_samp] / inner_F);
                    aver_weights[k] += (inner_weights[k] - prev_x2[k]) / inner_m;
                }
            }
            delete[] prev_x2;
            delete[] prev_x;
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
        delete[] grad_core_table;
        delete[] hyper_c_table;
    }
    delete[] full_grad;
    delete[] inner_weights;
    if(is_store_result)
        return stored_F;
    return NULL;
}

std::vector<double>* grad_desc_acc_dense::SGD_SCP2(double* X, double* Y, size_t N, blackbox* model
    , size_t iteration_no, double L, double step_size, bool is_store_result) {
    // Random Generator
    std::random_device rd;
    std::default_random_engine generator(rd());
    std::uniform_int_distribution<int> distribution(0, N - 1);
    // std::vector<double>* stored_weights = new std::vector<double>;
    std::vector<double>* stored_F = new std::vector<double>;
    double* inner_weights = new double[MAX_DIM];
    double* full_grad = new double[MAX_DIM];
    double* full_grad_core = new double[N];
    int regular = model->get_regularizer();
    double* lambda = model->get_params();
    size_t total_iterations = 0;
    copy_vec(inner_weights, model->get_model());
    // Init Weight Evaluate
    if(is_store_result)
        stored_F->push_back(model->zero_oracle_dense(X, Y, N));

    // Full Gradient
    for(size_t j = 0; j < N; j ++) {
        full_grad_core[j] = model->first_component_oracle_core_dense(X, Y, N, j);
        for(size_t k = 0; k < MAX_DIM; k ++) {
            full_grad[k] += X[j * MAX_DIM + k] * full_grad_core[j] / (double) N;
        }
    }
    // Stochastic Hyperplane Coeffs

    size_t n_hyperplane = 0;
    int* Ih = new int[N];
    std::vector<int>* Ir = new std::vector<int>;
    double ro = 0;
    double* _pR = new double[MAX_DIM];
    regularizer::first_oracle(regular, _pR, lambda, model->get_model());
    for(size_t j = 0; j < N; j ++) {
        Ih[j] = -1;
        double inner_Ff = 0;
        for(size_t k = 0; k < MAX_DIM; k ++)
            inner_Ff += (full_grad[k] + _pR[k])
                    * (X[j * MAX_DIM + k] * full_grad_core[j] + _pR[k]);
        if(inner_Ff > ro) {
            Ir->push_back(j);
            Ih[j] = n_hyperplane ++;
        }
    }
    delete[] _pR;
    double* grad_core_table = new double[n_hyperplane * MAX_DIM];
    double* hyper_c_table = new double[n_hyperplane];
    memset(grad_core_table, 0, n_hyperplane * MAX_DIM * sizeof(double));
    memset(hyper_c_table, 0, n_hyperplane * sizeof(double));

    for(size_t j = 0; j < n_hyperplane; j ++) {
        for(size_t k = 0; k < MAX_DIM; k ++) {
            grad_core_table[j * MAX_DIM + k] = X[(*Ir)[j] * MAX_DIM + k] * full_grad_core[(*Ir)[j]] + _pR[k];
            hyper_c_table[j] += model->get_model()[k] * grad_core_table[j * MAX_DIM + k];
        }
    }
    // OUTTER_LOOP
    for(size_t i = 0 ; i < iteration_no; i ++) {
        // INNER_LOOP
        for(size_t j = 0; j < N; j ++) {
            int rand_samp = distribution(generator);
            double inner_core = model->first_component_oracle_core_dense(X, Y, N
                , rand_samp, inner_weights);
            double hyper_x = 0;
            double* prev_x = new double[MAX_DIM];
            int ix = Ih[rand_samp];
            if(ix == -1) ix = 100;
            for(size_t k = 0; k < MAX_DIM; k ++) {
                double val = X[rand_samp * MAX_DIM + k];
                double sub_grad = inner_core * val;
                prev_x[k] = inner_weights[k];
                inner_weights[k] -= step_size * sub_grad;
                regularizer::proximal_operator(regular, inner_weights[k], step_size, lambda);
                hyper_x += inner_weights[k] * grad_core_table[ix * MAX_DIM + k];

            }
            if(Ih[rand_samp] != -1 && hyper_x < hyper_c_table[ix]) {
                double hyper_c = 0;
                double* _pR = new double[MAX_DIM];
                regularizer::first_oracle(regular, _pR, lambda, prev_x);
                for(size_t k = 0; k < MAX_DIM; k ++) {
                    double val = X[rand_samp * MAX_DIM + k];
                    double sub_grad = inner_core * val + _pR[k];
                    hyper_c += prev_x[k] * sub_grad;
                    grad_core_table[ix * MAX_DIM + k] = sub_grad;
                }
                hyper_c_table[ix] = hyper_c;
                delete[] _pR;
            }
            else if(hyper_x > hyper_c_table[ix]) {
                double inner_F = 0;
                for(size_t k = 0; k < MAX_DIM; k ++) {
                    inner_F += grad_core_table[ix * MAX_DIM + k]
                            * grad_core_table[ix * MAX_DIM + k];
                }
                for(size_t k = 0; k < MAX_DIM; k ++) {
                    inner_weights[k] = inner_weights[k] - grad_core_table[ix * MAX_DIM + k]
                                * (hyper_x / inner_F - hyper_c_table[ix] / inner_F);
                }
            }
            delete[] prev_x;
            total_iterations ++;
        }
        model->update_model(inner_weights);
        // For Matlab (per m/n passes)
        if(is_store_result) {
            stored_F->push_back(model->zero_oracle_dense(X, Y, N));
        }
    }
    delete[] full_grad_core;
    delete[] grad_core_table;
    delete[] hyper_c_table;
    delete[] full_grad;
    delete[] inner_weights;
    if(is_store_result)
        return stored_F;
    return NULL;
}

// Not Done
std::vector<double>* grad_desc_acc_dense::Katyusha_plus(double* X, double* Y, size_t N, blackbox* model
    , size_t iteration_no, double L, double sigma, double step_size, bool is_store_result) {
    // Random Generator
    std::vector<double>* stored_F = new std::vector<double>;
    std::random_device rd;
    std::default_random_engine generator(rd());
    std::uniform_int_distribution<int> distribution(0, N - 1);
    size_t m = 2.0 * N;
    size_t total_iterations = 0;
    double tau_2 = 0.5, tau_1 = 0.5;
    if(sqrt(sigma * m / (3.0 * L)) < 0.5) tau_1 = sqrt(sigma * m / (3.0 * L));
    double alpha = 1.0 / (tau_1 * 3.0 * L);
    int regular = model->get_regularizer();
    double* lambda = model->get_params();
    double step_size_y = 1.0 / (3.0 * L);
    double compos_factor = 1.0 + alpha * sigma;
    double compos_base = (pow((double)compos_factor, (double)m) - 1.0) / (alpha * sigma);
    double* compos_pow = new double[m + 1];
    for(size_t i = 0; i <= m; i ++)
        compos_pow[i] = pow((double)compos_factor, (double)i);
    double* y = new double[MAX_DIM];
    double* z = new double[MAX_DIM];
    double* inner_weights = new double[MAX_DIM];
    double* full_grad = new double[MAX_DIM];
    // init vectors
    copy_vec(y, model->get_model());
    copy_vec(z, model->get_model());
    copy_vec(inner_weights, model->get_model());
    // Init Weight Evaluate
    if(is_store_result)
        stored_F->push_back(model->zero_oracle_dense(X, Y, N));
    // OUTTER LOOP
    for(size_t i = 0; i < iteration_no; i ++) {
        double* full_grad_core = new double[N];
        double* outter_weights = (model->get_model());
        double* aver_weights = new double[MAX_DIM];
        memset(full_grad, 0, MAX_DIM * sizeof(double));
        memset(aver_weights, 0, MAX_DIM * sizeof(double));
        switch(regular) {
            case regularizer::L2:
            case regularizer::ELASTIC_NET: // Strongly Convex Case
                break;
            case regularizer::L1: // Non-Strongly Convex Case
                tau_1 = 2.0 / ((double) i + 4.0);
                alpha = 1.0 / (tau_1 * 3.0 * L);
                break;
            default:
                throw std::string("500 Internal Error.");
                break;
        }
        // Full Gradient
        for(size_t j = 0; j < N; j ++) {
            full_grad_core[j] = model->first_component_oracle_core_dense(X, Y, N, j);
            for(size_t k = 0; k < MAX_DIM; k ++) {
                full_grad[k] += X[j * MAX_DIM + k] * full_grad_core[j] / (double) N;
            }
        }
        // 0th Inner Iteration
        for(size_t k = 0; k < MAX_DIM; k ++)
            inner_weights[k] = tau_1 * z[k] + tau_2 * outter_weights[k]
                             + (1 - tau_1 - tau_2) * y[k];
        // INNER LOOP
        for(size_t j = 0; j < m; j ++) {
            int rand_samp = distribution(generator);
            double inner_core = model->first_component_oracle_core_dense(X, Y, N, rand_samp, inner_weights);
            for(size_t k = 0; k < MAX_DIM; k ++) {
                double val = X[rand_samp * MAX_DIM + k];
                double katyusha_grad = full_grad[k] + val * (inner_core - full_grad_core[rand_samp]);
                double prev_z = z[k];
                z[k] -= alpha * katyusha_grad;
                regularizer::proximal_operator(regular, z[k], alpha, lambda);
                ////// For Katyusha With Update Option I //////
                // y[k] = inner_weights[k] - step_size_y * katyusha_grad;
                // regularizer::proximal_operator(regular, y[k], step_size_y, lambda);
                ////// For Katyusha With Update Option II //////
                y[k] = inner_weights[k] + tau_1 * (z[k] - prev_z);

                switch(regular) {
                    case regularizer::L2:
                    case regularizer::ELASTIC_NET: // Strongly Convex Case
                        aver_weights[k] += compos_pow[j] / compos_base * y[k];
                        break;
                    case regularizer::L1: // Non-Strongly Convex Case
                        aver_weights[k] += y[k] / m;
                        break;
                    default:
                        throw std::string("500 Internal Error.");
                        break;
                }
                // (j + 1)th Inner Iteration
                if(j < m - 1)
                    inner_weights[k] = tau_1 * z[k] + tau_2 * outter_weights[k]
                                     + (1 - tau_1 - tau_2) * y[k];
            }
            total_iterations ++;
        }
        model->update_model(aver_weights);
        delete[] aver_weights;
        delete[] full_grad_core;
        // For Matlab
        if(is_store_result) {
            stored_F->push_back(model->zero_oracle_dense(X, Y, N));
        }
    }
    delete[] y;
    delete[] z;
    delete[] inner_weights;
    delete[] full_grad;
    delete[] compos_pow;
    if(is_store_result)
        return stored_F;
    return NULL;
}
