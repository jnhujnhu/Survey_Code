#include "grad_desc_sparse.hpp"
#include "utils.hpp"
#include "regularizer.hpp"
#include <random>
#include <cmath>
#include <string.h>

extern size_t MAX_DIM;

double* grad_desc_sparse::GD(double* X, double* Y, size_t* Jc, size_t* Ir
    , size_t N, blackbox* model, size_t iteration_no, double L, double step_size
    , bool is_store_weight, bool is_debug_mode, bool is_store_result) {
    // double* stored_weights = NULL;
    double* stored_F = NULL;
    size_t passes = iteration_no;
    // if(is_store_weight)
    //     stored_weights = new double[iteration_no * MAX_DIM];
    if(is_store_result)
        stored_F = new double[passes];
    double* new_weights = new double[MAX_DIM];
    for(size_t i = 0; i < iteration_no; i ++) {
        double* full_grad = new double[MAX_DIM];
        memset(full_grad, 0, MAX_DIM * sizeof(double));
        for(size_t j = 0; j < N; j ++) {
            double full_grad_core = model->first_component_oracle_core_sparse(X, Y, Jc, Ir, N, j);
            for(size_t k = Jc[j]; k < Jc[j + 1]; k ++) {
                full_grad[Ir[k]] += X[k] * full_grad_core / (double) N;
            }
        }
        double* _pR = new double[MAX_DIM];
        model->first_regularizer_oracle(_pR);
        for(size_t j = 0; j < MAX_DIM; j ++) {
            full_grad[j] += _pR[j];
            new_weights[j] = (model->get_model())[j] - step_size * (full_grad[j]);
        }
        delete[] _pR;
        model->update_model(new_weights);
        if(is_debug_mode) {
            double log_F = log(model->zero_oracle_sparse(X, Y, Jc, Ir, N));
            printf("GD: Iteration %zd, log_F: %lf.\n", i, log_F);
        }
        else
            printf("GD: Iteration %zd.\n", i);
        // For Drawing
        // if(is_store_weight) {
        //     for(size_t j = 0; j < MAX_DIM; j ++) {
        //         stored_weights[i * MAX_DIM + j] = new_weights[j];
        //     }
        // }
        //For Matlab
        if(is_store_result) {
            stored_F[i] = model->zero_oracle_sparse(X, Y, Jc, Ir, N);
        }
        delete[] full_grad;
    }
    //Final Output
    double log_F = log(model->zero_oracle_sparse(X, Y, Jc, Ir, N));
    printf("GD: Iteration %zd, log_F: %lf.\n", iteration_no, log_F);
    delete[] new_weights;
    // if(is_store_weight)
    //     return stored_weights;
    if(is_store_result)
        return stored_F;
    return NULL;
}

double* grad_desc_sparse::SGD(double* X, double* Y, size_t* Jc, size_t* Ir
    , size_t N, blackbox* model, size_t iteration_no, double L, double step_size
    , bool is_store_weight, bool is_debug_mode, bool is_store_result) {
    // Random Generator
    std::random_device rd;
    std::default_random_engine generator(rd());
    std::uniform_int_distribution<int> distribution(0, N - 1);
    double* stored_F = NULL;
    size_t passes = (size_t) floor((double) iteration_no / N);
    int regular = model->get_regularizer();
    double* lambda = model->get_params();
    // double* stored_weights = NULL;
    // lazy updates extra array.
    int* last_seen = new int[MAX_DIM];
    for(size_t i = 0; i < MAX_DIM; i ++) last_seen[i] = -1;
    // For Drawing
    // if(is_store_weight)
    //     stored_weights = new double[iteration_no * MAX_DIM];
    // For Matlab
    if(is_store_result) {
        stored_F = new double[passes + 1];
        stored_F[0] = model->zero_oracle_sparse(X, Y, Jc, Ir, N);
    }

    double* sub_grad = new double[MAX_DIM];
    double* new_weights = new double[MAX_DIM];
    copy_vec(new_weights, model->get_model());
    for(size_t i = 0; i < iteration_no; i ++) {
        int rand_samp = distribution(generator);
        double core = model->first_component_oracle_core_sparse(X, Y, Jc, Ir, N, rand_samp, new_weights);
        for(size_t j = Jc[rand_samp]; j < Jc[rand_samp + 1]; j ++) {
            size_t index = Ir[j];
            // lazy update.
            if((int)i > last_seen[index] + 1) {
                regularizer::proximal_operator(regular, new_weights[index], step_size, lambda
                    , i - (last_seen[index] + 1), false);
            }
            new_weights[index] -= step_size * core * X[j];
            regularizer::proximal_operator(regular, new_weights[index], step_size, lambda);
            last_seen[index] = i;
        }
        // if(is_debug_mode) {
        //     double log_F = log(model->zero_oracle_sparse(X, Y, Jc, Ir, N));
        //     printf("SGD: Iteration %zd, log_F: %lf.\n", i, log_F);
        // }
        // For Drawing
        // if(is_store_weight) {
        //     for(size_t j = 0; j < MAX_DIM; j ++) {
        //         stored_weights[i * MAX_DIM + j] = new_weights[j];
        //     }
        // }
        // For Matlab
        if(is_store_result) {
            if(!((i + 1) % N)) {
                for(size_t j = 0; j < MAX_DIM; j ++) {
                    if((int)i > last_seen[j]) {
                        regularizer::proximal_operator(regular, new_weights[j], step_size, lambda
                            , i - last_seen[j], false);
                        last_seen[j] = i;
                    }
                }
                stored_F[(size_t) floor((double) i / N) + 1]
                    = model->zero_oracle_sparse(X, Y, Jc, Ir, N, new_weights);
            }
        }
    }
    // lazy update aggragate
    for(size_t i = 0; i < MAX_DIM; i ++) {
        if((int)iteration_no > last_seen[i] + 1) {
            regularizer::proximal_operator(regular, new_weights[i], step_size, lambda
                , iteration_no - (last_seen[i] + 1), false);
        }
    }
    model->update_model(new_weights);
    //Final Output
    // double log_F = log(model->zero_oracle_sparse(X, Y, Jc, Ir, N));
    // printf("SGD: Iteration %zd, log_F: %lf.\n", iteration_no, log_F);
    delete[] sub_grad;
    delete[] new_weights;
    delete[] last_seen;
    // For Drawing
    // if(is_store_weight)
    //     return stored_weights;
    // For Matlab
    if(is_store_result)
        return stored_F;
    return NULL;
}

std::vector<double>* grad_desc_sparse::Prox_SVRG(double* X, double* Y, size_t* Jc, size_t* Ir
    , size_t N, blackbox* model, size_t iteration_no, int Mode, double L, double step_size
    , bool is_store_weight, bool is_debug_mode, bool is_store_result) {
    // Random Generator
    std::random_device rd;
    std::default_random_engine generator(rd());
    std::uniform_int_distribution<int> distribution(0, N - 1);
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
        stored_F->push_back(model->zero_oracle_sparse(X, Y, Jc, Ir, N));
    // OUTTER_LOOP
    for(size_t i = 0 ; i < iteration_no; i ++) {
        double* full_grad_core = new double[N];
        // Average Iterates
        double* aver_weights = new double[MAX_DIM];
        // lazy update extra params.
        int* last_seen = new int[MAX_DIM];
        //FIXME: SVRG / SVRG++
        double inner_m = m0;//pow(2, i + 1) * m0;
        memset(aver_weights, 0, MAX_DIM * sizeof(double));
        memset(full_grad, 0, MAX_DIM * sizeof(double));
        for(size_t i = 0; i < MAX_DIM; i ++) last_seen[i] = -1;
        // Full Gradient
        for(size_t j = 0; j < N; j ++) {
            full_grad_core[j] = model->first_component_oracle_core_sparse(X, Y, Jc, Ir, N, j);
            for(size_t k = Jc[j]; k < Jc[j + 1]; k ++) {
                full_grad[Ir[k]] += X[k] * full_grad_core[j] / (double) N;
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
            double inner_core = model->first_component_oracle_core_sparse(X, Y
                        , Jc, Ir, N, rand_samp, inner_weights);
            for(size_t k = Jc[rand_samp]; k < Jc[rand_samp + 1]; k ++) {
                size_t index = Ir[k];
                double val = X[k];
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
                double vr_sub_grad = (inner_core - full_grad_core[rand_samp]) * val + full_grad[index];
                inner_weights[index] -= step_size * vr_sub_grad;
                aver_weights[index] += regularizer::proximal_operator(regular, inner_weights[index]
                    , step_size, lambda) / inner_m;
                last_seen[index] = j;
            }
            total_iterations ++;
            // For Drawing
            // if(is_store_weight) {
            //     for(size_t k = 0; k < MAX_DIM; k ++)
            //         stored_weights->push_back(inner_weights[k]);
            // }
            // if(is_debug_mode) {
            //     double log_F = log(model->zero_oracle(data, inner_weights));
            //     printf("Prox_SVRG: Outter Iteration: %zd -> Inner Iteration %zd, log_F for inner_weights: %lf.\n", i, j, log_F);
            // }
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
        delete[] full_grad_core;
        // if(is_debug_mode) {
        //     double log_F = log(model->zero_oracle_sparse(X, Y, Jc, Ir, N));
        //     printf("Prox_SVRG: Outter Iteration %zd, log_F: %lf.\n", i, log_F);
        // }
    }
    delete[] full_grad;
    delete[] inner_weights;
    //Final Output
    // double log_F = log(model->zero_oracle_sparse(X, Y, Jc, Ir, N));
    // printf("Prox_SVRG: Total Iteration No.: %zd, logF = %lf.\n", total_iterations, log_F);
    // if(is_store_weight)
    //     return stored_weights;
    if(is_store_result)
        return stored_F;
    return NULL;
}

// w = A*w + Const
double lazy_update_SVRG(double& w, double Const, double A, size_t times, bool is_averaged = true) {
    if(times == 1) {
        w = A * w + Const;
        return w;
    }
    double pow_A = pow((double)A, (double)times);
    double T1 = equal_ratio(A, pow_A, times);
    double lazy_average = 0.0;
    if(is_averaged) {
        double T2 = Const / (1 - A);
        lazy_average = T1 * w + T2 * times - T1 * T2;
    }
    w = pow_A * w + Const * T1 / A;
    return lazy_average;
}

// Only Applicable for L2 regularizer
std::vector<double>* grad_desc_sparse::SVRG(double* X, double* Y, size_t* Jc, size_t* Ir
    , size_t N, blackbox* model, size_t iteration_no, int Mode, double L
    , double step_size, bool is_store_weight, bool is_debug_mode, bool is_store_result) {
        // Random Generator
        std::random_device rd;
        std::default_random_engine generator(rd());
        std::uniform_int_distribution<int> distribution(0, N - 1);
        // std::vector<double>* stored_weights = new std::vector<double>;
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
            stored_F->push_back(model->zero_oracle_sparse(X, Y, Jc, Ir, N));
        // OUTTER_LOOP
        for(size_t i = 0 ; i < iteration_no; i ++) {
            double* full_grad_core = new double[N];
            // Average Iterates
            double* aver_weights = new double[MAX_DIM];
            // lazy update extra params.
            int* last_seen = new int[MAX_DIM];
            //FIXME: SVRG / SVRG++
            double inner_m = m0;//pow(2, i + 1) * m0;
            memset(aver_weights, 0, MAX_DIM * sizeof(double));
            memset(full_grad, 0, MAX_DIM * sizeof(double));
            for(size_t i = 0; i < MAX_DIM; i ++) last_seen[i] = -1;
            // Full Gradient
            for(size_t j = 0; j < N; j ++) {
                full_grad_core[j] = model->first_component_oracle_core_sparse(X, Y, Jc, Ir, N, j);
                for(size_t k = Jc[j]; k < Jc[j + 1]; k ++) {
                    full_grad[Ir[k]] += (X[k] * full_grad_core[j]) / (double) N;
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
            for(size_t j = 0; j < inner_m ; j ++) {
                int rand_samp = distribution(generator);
                double inner_core = model->first_component_oracle_core_sparse(X, Y, Jc, Ir, N
                    , rand_samp, inner_weights);
                for(size_t k = Jc[rand_samp]; k < Jc[rand_samp + 1]; k ++) {
                    size_t index = Ir[k];
                    double val = X[k];
                    // lazy update
                    if((int)j > last_seen[index] + 1) {
                        switch(Mode) {
                            case SVRG_LAST_LAST:
                                lazy_update_SVRG(inner_weights[index], -step_size * full_grad[index]
                                    , 1 - step_size * lambda[0], j - (last_seen[index] + 1), false);
                                break;
                            case SVRG_AVER_LAST:
                            case SVRG_AVER_AVER:
                                aver_weights[index] += lazy_update_SVRG(inner_weights[index]
                                    , -step_size * full_grad[index], 1 - step_size * lambda[0]
                                    , j - (last_seen[index] + 1)) / inner_m;
                                break;
                            default:
                                throw std::string("500 Internal Error.");
                                break;
                        }
                    }
                    double vr_sub_grad = (inner_core - full_grad_core[rand_samp]) * val
                             + inner_weights[index]* lambda[0] + full_grad[index];
                    inner_weights[index] -= step_size * vr_sub_grad;
                    aver_weights[index] += inner_weights[index] / inner_m;
                    last_seen[index] = j;
                }
                total_iterations ++;
                // For Drawing
                // if(is_store_weight) {
                //     for(size_t k = 0; k < MAX_DIM; k ++)
                //         stored_weights->push_back(inner_weights[k]);
                // }
                // if(is_debug_mode) {
                //     double log_F = log(model->zero_oracle_sparse(X, Y, Jc, Ir, N, inner_weights));
                //     printf("SVRG: Outter Iteration: %zd -> Inner Iteration %zd, log_F for inner_weights: %lf.\n", i, j, log_F);
                // }
            }
            // lazy update aggragate
            switch(Mode) {
                case SVRG_LAST_LAST:
                    for(size_t j = 0; j < MAX_DIM; j ++) {
                        if(inner_m > last_seen[j] + 1) {
                            lazy_update_SVRG(inner_weights[j], -step_size * full_grad[j]
                                , 1 - step_size * lambda[0], inner_m - (last_seen[j] + 1), false);
                        }
                    }
                    model->update_model(inner_weights);
                    break;
                case SVRG_AVER_LAST:
                case SVRG_AVER_AVER:
                    for(size_t j = 0; j < MAX_DIM; j ++) {
                        if(inner_m > last_seen[j] + 1) {
                            aver_weights[j] += lazy_update_SVRG(inner_weights[j]
                                , -step_size * full_grad[j], 1 - step_size * lambda[0]
                                , inner_m - (last_seen[j] + 1)) / inner_m;
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
            delete[] full_grad_core;
            // if(is_debug_mode) {
            //     double log_F = log(model->zero_oracle_sparse(X, Y, Jc, Ir, N));
            //     printf("SVRG: Outter Iteration %zd, log_F: %lf.\n", i, log_F);
            // }
        }
        delete[] full_grad;
        delete[] inner_weights;
        //Final Output
        // double log_F = log(model->zero_oracle_sparse(X, Y, Jc, Ir, N));
        // printf("SVRG: Total Iteration No.: %zd, logF = %lf.\n", total_iterations, log_F);
        // if(is_store_weight)
        //     return stored_weights;
        if(is_store_result)
            return stored_F;
        return NULL;
}

// Magic Code
double Katyusha_Y_L2_proximal(double& _y0, double _z0, double tau_1, double tau_2
    , double lambda, double step_size_y, double alpha, double _outterx, double _F
    , size_t times, int start_iter, double compos_factor, double compos_base, double* compos_pow) {
    double lazy_average = 0.0;
    // Constants
    double prox_y = 1.0 / (1.0 + step_size_y * lambda);
    double ETA = (1.0 - tau_1 - tau_2) * prox_y;
    double M = tau_1 * prox_y * (_z0 + _F / lambda);
    double A = 1.0 / (1.0 + alpha * lambda);
    double constant = prox_y * (tau_2 * _outterx - _F * (step_size_y + tau_1 / lambda));
    double MAETA = M / (A - ETA), CONSTETA = constant / (1.0 - ETA);
    // Powers
    double pow_eta = pow((double) ETA, (double) times);
    double pow_A = pow((double) A, (double) times);

    if(start_iter == -1)
        lazy_average = 1.0 / (compos_base * compos_factor);
    else
        lazy_average = compos_pow[start_iter] / compos_base;
    lazy_average *= equal_ratio(ETA * compos_factor, pow_eta * compos_pow[times], times)* (_y0 - MAETA - CONSTETA)
                 + equal_ratio(A * compos_factor, pow_A * compos_pow[times], times)* MAETA
                 + equal_ratio(compos_factor, compos_pow[times], times) * CONSTETA;

    _y0 = pow_eta * (_y0 - MAETA - CONSTETA) + MAETA * pow_A + CONSTETA;
    return lazy_average;
}

std::vector<double>* grad_desc_sparse::Katyusha(double* X, double* Y, size_t* Jc, size_t* Ir
    , size_t N, blackbox* model, size_t iteration_no, double L, double sigma
    , double step_size, bool is_store_weight, bool is_debug_mode, bool is_store_result) {
    // Random Generator
    std::vector<double>* stored_F = new std::vector<double>;
    std::random_device rd;
    std::default_random_engine generator(rd());
    std::uniform_int_distribution<int> distribution(0, N - 1);
    size_t m = 2.0 * N;
    size_t total_iterations = 0;
    int regular = model->get_regularizer();
    double* lambda = model->get_params();
    double tau_2 = 0.5, tau_1 = 0.5;
    if(sqrt(sigma * m / (3.0 * L)) < 0.5) tau_1 = sqrt(sigma * m / (3.0 * L));
    double alpha = 1.0 / (tau_1 * 3.0 * L);
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
        stored_F->push_back(model->zero_oracle_sparse(X, Y, Jc, Ir, N));
    // OUTTER LOOP
    for(size_t i = 0; i < iteration_no; i ++) {
        double* full_grad_core = new double[N];
        double* outter_weights = (model->get_model());
        double* aver_weights = new double[MAX_DIM];
        // lazy update extra param
        double* last_seen = new double[MAX_DIM];
        memset(full_grad, 0, MAX_DIM * sizeof(double));
        memset(aver_weights, 0, MAX_DIM * sizeof(double));
        for(size_t i = 0; i < MAX_DIM; i ++) last_seen[i] = -1;
        // Full Gradient
        for(size_t j = 0; j < N; j ++) {
            full_grad_core[j] = model->first_component_oracle_core_sparse(X, Y, Jc, Ir, N, j);
            for(size_t k = Jc[j]; k < Jc[j + 1]; k ++) {
                full_grad[Ir[k]] += X[k] * full_grad_core[j] / (double) N;
            }
        }
        // 0th Inner Iteration
        for(size_t k = 0; k < MAX_DIM; k ++)
            inner_weights[k] = tau_1 * z[k] + tau_2 * outter_weights[k]
                             + (1 - tau_1 - tau_2) * y[k];
        // INNER LOOP
        for(size_t j = 0; j < m; j ++) {
            int rand_samp = distribution(generator);
            double inner_core = model->first_component_oracle_core_sparse(X, Y, Jc, Ir, N, rand_samp, inner_weights);
            for(size_t k = Jc[rand_samp]; k < Jc[rand_samp + 1]; k ++) {
                size_t index = Ir[k];
                double val = X[k];
                // lazy update
                if((int)j > last_seen[index] + 1) {
                    aver_weights[index] += Katyusha_Y_L2_proximal(y[index], z[index]
                        , tau_1, tau_2, lambda[0], step_size_y, alpha, outter_weights[index]
                        , full_grad[index], j - (last_seen[index] + 1), last_seen[index]
                        , compos_factor, compos_base, compos_pow);
                    regularizer::proximal_operator(regular, z[index], alpha, lambda, j - (last_seen[index] + 1), false
                         , -alpha * full_grad[index]);
                    inner_weights[index] = tau_1 * z[index] + tau_2 * outter_weights[index]
                                     + (1 - tau_1 - tau_2) * y[index];
                }
                double katyusha_grad = full_grad[index] + val * (inner_core - full_grad_core[rand_samp]);
                z[index] -= alpha * katyusha_grad;
                regularizer::proximal_operator(regular, z[index], alpha, lambda);
                y[index] = inner_weights[index] - step_size_y * katyusha_grad;
                regularizer::proximal_operator(regular, y[index], step_size_y, lambda);
                aver_weights[index] += compos_pow[j] / compos_base * y[index];
                // (j + 1)th Inner Iteration
                if(j < m - 1)
                    inner_weights[index] = tau_1 * z[index] + tau_2 * outter_weights[index]
                                     + (1 - tau_1 - tau_2) * y[index];
                last_seen[index] = j;
            }
            total_iterations ++;
            // if(is_debug_mode) {
            //     double log_F = log(model->zero_oracle_sparse(X, Y, Jc, Ir, N, inner_weights));
            //     printf("Katyusha: Outter Iteration: %zd -> Inner Iteration %zd, log_F for inner_weights: %lf.\n", i, j, log_F);
            // }
        }
        // lazy update aggragate
        for(size_t j = 0; j < MAX_DIM; j ++) {
            if(m > last_seen[j] + 1) {
                aver_weights[j] += Katyusha_Y_L2_proximal(y[j], z[j]
                    , tau_1, tau_2, lambda[0], step_size_y, alpha, outter_weights[j]
                    , full_grad[j], m - (last_seen[j] + 1), last_seen[j]
                    , compos_factor, compos_base, compos_pow);
                regularizer::proximal_operator(regular, z[j], alpha, lambda, m - (last_seen[j] + 1), false
                     , -alpha * full_grad[j]);
                inner_weights[j] = tau_1 * z[j] + tau_2 * outter_weights[j]
                                 + (1 - tau_1 - tau_2) * y[j];
            }
        }
        model->update_model(aver_weights);
        delete[] aver_weights;
        delete[] full_grad_core;
        delete[] last_seen;
        // For Matlab
        if(is_store_result) {
            stored_F->push_back(model->zero_oracle_sparse(X, Y, Jc, Ir, N));
        }
        // if(is_debug_mode) {
        //     double log_F = log(model->zero_oracle_sparse(X, Y, Jc, Ir, N));
        //     printf("Katyusha: Outter Iteration %zd, log_F: %lf.\n", i, log_F);
        // }
    }
    delete[] y;
    delete[] z;
    delete[] inner_weights;
    delete[] full_grad;
    delete[] compos_pow;
    //Final Output
    // double log_F = log(model->zero_oracle_sparse(X, Y, Jc, Ir, N));
    // printf("Katyusha: Total Iteration No.: %zd, log_F: %lf.\n", total_iterations, log_F);
    if(is_store_result)
        return stored_F;
    return NULL;
}

std::vector<double>* grad_desc_sparse::SAGA(double* X, double* Y, size_t* Jc, size_t* Ir, size_t N
    , blackbox* model, size_t iteration_no, double L, double step_size, bool is_store_weight
    , bool is_debug_mode, bool is_store_result) {
        // Random Generator
    std::random_device rd;
    std::default_random_engine generator(rd());
    std::uniform_int_distribution<int> distribution(0, N - 1);
    std::vector<double>* stored_F = new std::vector<double>;
    int regular = model->get_regularizer();
    double* lambda = model->get_params();
    // For Matlab
    if(is_store_result) {
        stored_F->push_back(model->zero_oracle_sparse(X, Y, Jc, Ir, N));
        // Extra Pass for Create Gradient Table
        stored_F->push_back((*stored_F)[0]);
    }
    double* new_weights = new double[MAX_DIM];
    double* grad_core_table = new double[N];
    double* aver_grad = new double[MAX_DIM];
    int* last_seen = new int[MAX_DIM];
    memset(aver_grad, 0, MAX_DIM * sizeof(double));
    for(size_t j = 0; j < MAX_DIM; j ++) last_seen[j] = -1;

    copy_vec(new_weights, model->get_model());
    // Init Gradient Core Table
    for(size_t i = 0; i < N; i ++) {
        grad_core_table[i] = model->first_component_oracle_core_sparse(X, Y, Jc, Ir, N, i);
        for(size_t j = Jc[i]; j < Jc[i + 1]; j ++)
            aver_grad[Ir[j]] += grad_core_table[i] * X[j] / N;
    }

    size_t skip_pass = 0;
    for(size_t i = 0; i < iteration_no; i ++) {
        int rand_samp = distribution(generator);
        double core = model->first_component_oracle_core_sparse(X, Y, Jc, Ir, N, rand_samp, new_weights);
        double past_grad_core = grad_core_table[rand_samp];
        grad_core_table[rand_samp] = core;
        for(size_t j = Jc[rand_samp]; j < Jc[rand_samp + 1]; j ++) {
            size_t index = Ir[j];
            // lazy update or lagged update in Schmidt et al.[2016]
            if((int) i > last_seen[index] + 1) {
                regularizer::proximal_operator(regular, new_weights[index], step_size
                        , lambda, i - (last_seen[index] + 1), false, -step_size * aver_grad[index]);
            }
            // Update Weight
            new_weights[index] -= step_size * ((core - past_grad_core)* X[j] + aver_grad[index]);
            // Update Gradient Table Average
            aver_grad[index] -= (past_grad_core - core) * X[j] / N;
            regularizer::proximal_operator(regular, new_weights[index], step_size, lambda);
            last_seen[index] = i;
        }
        // For Matlab
        if(is_store_result) {
            if(!(i % N)) {
                skip_pass ++;
                if(skip_pass == 3) {
                    // Force Lazy aggragate for function value evaluate
                    for(size_t j = 0; j < MAX_DIM; j ++) {
                        if((int)i > last_seen[j]) {
                            regularizer::proximal_operator(regular, new_weights[j], step_size, lambda
                                , i - last_seen[j], false, -step_size * aver_grad[j]);
                            last_seen[j] = i;
                        }
                    }
                    stored_F->push_back(model->zero_oracle_sparse(X, Y, Jc, Ir, N, new_weights));
                    skip_pass = 0;
                }
            }
        }
    }
    // lazy update aggragate
    for(size_t i = 0; i < MAX_DIM; i ++) {
        if((int)iteration_no > last_seen[i] + 1) {
            regularizer::proximal_operator(regular, new_weights[i], step_size, lambda
                , iteration_no - (last_seen[i] + 1), false, -step_size * aver_grad[i]);
        }
    }
    model->update_model(new_weights);
    delete[] new_weights;
    delete[] grad_core_table;
    delete[] aver_grad;
    // For Matlab
    if(is_store_result)
        return stored_F;
    return NULL;
}
