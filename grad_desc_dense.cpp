#include "grad_desc_dense.hpp"
#include "utils.hpp"
#include "regularizer.hpp"
#include <random>
#include <cmath>
#include <string.h>

extern size_t MAX_DIM;

double* grad_desc_dense::GD(double* X, double* Y, size_t N, blackbox* model, size_t iteration_no
    , double L, double step_size, bool is_store_weight, bool is_debug_mode, bool is_store_result) {
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
            double full_grad_core = model->first_component_oracle_core_dense(X, Y, N, j);
            for(size_t k = 0; k < MAX_DIM; k ++) {
                full_grad[k] += X[j * MAX_DIM + k] * full_grad_core / (double) N;
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
            double log_F = log(model->zero_oracle_dense(X, Y, N));
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
            stored_F[i] = model->zero_oracle_dense(X, Y, N);
        }
        delete[] full_grad;
    }
    //Final Output
    double log_F = log(model->zero_oracle_dense(X, Y, N));
    printf("GD: Iteration %zd, log_F: %lf.\n", iteration_no, log_F);
    delete[] new_weights;
    // if(is_store_weight)
    //     return stored_weights;
    if(is_store_result)
        return stored_F;
    return NULL;
}

double* grad_desc_dense::SGD(double* X, double* Y, size_t N, blackbox* model, size_t iteration_no
    , double L, double step_size, bool is_store_weight, bool is_debug_mode, bool is_store_result) {
    // Random Generator
    std::random_device rd;
    std::default_random_engine generator(rd());
    std::uniform_int_distribution<int> distribution(0, N - 1);
    double* stored_F = NULL;
    size_t passes = (size_t) floor((double) iteration_no / N);
    int regular = model->get_regularizer();
    double* lambda = model->get_params();
    // double* stored_weights = NULL;
    // For Drawing
    // if(is_store_weight)
    //     stored_weights = new double[iteration_no * MAX_DIM];
    // For Matlab
    if(is_store_result) {
        stored_F = new double[passes + 1];
        stored_F[0] = model->zero_oracle_dense(X, Y, N);
    }
    double* new_weights = new double[MAX_DIM];
    copy_vec(new_weights, model->get_model());
    for(size_t i = 0; i < iteration_no; i ++) {
        int rand_samp = distribution(generator);
        double core = model->first_component_oracle_core_dense(X, Y, N, rand_samp, new_weights);
        for(size_t j = 0; j < MAX_DIM; j ++) {
            new_weights[j] -= step_size * core * X[rand_samp * MAX_DIM + j];
            regularizer::proximal_operator(regular, new_weights[j], step_size, lambda);
        }
        // if(is_debug_mode) {
        //     double log_F = log(model->zero_oracle_dense(X, Y, N));
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
                stored_F[(size_t) floor((double) i / N) + 1] = model->zero_oracle_dense(X, Y, N, new_weights);
            }
        }
    }
    model->update_model(new_weights);
    //Final Output
    // double log_F = log(model->zero_oracle_dense(X, Y, N));
    // printf("SGD: Iteration %zd, log_F: %lf.\n", iteration_no, log_F);
    delete[] new_weights;
    // For Drawing
    // if(is_store_weight)
    //     return stored_weights;
    // For Matlab
    if(is_store_result)
        return stored_F;
    return NULL;
}

std::vector<double>* grad_desc_dense::Prox_SVRG(double* X, double* Y, size_t N, blackbox* model
    , size_t iteration_no, int Mode, double L, double step_size, bool is_store_weight, bool is_debug_mode, bool is_store_result) {
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
        // if(is_debug_mode) {
        //     double log_F = log(model->zero_oracle_dense(X, Y, N));
        //     printf("Prox_SVRG: Outter Iteration %zd, log_F: %lf.\n", i, log_F);
        // }
    }
    delete[] full_grad;
    delete[] inner_weights;
    //Final Output
    // double log_F = log(model->zero_oracle_dense(X, Y, N));
    // printf("Prox_SVRG: Total Iteration No.: %zd, logF = %lf.\n", total_iterations, log_F);
    // if(is_store_weight)
    //     return stored_weights;
    if(is_store_result)
        return stored_F;
    return NULL;
}

// Only Applicable for L2 regularizer
std::vector<double>* grad_desc_dense::SVRG(double* X, double* Y, size_t N, blackbox* model
    , size_t iteration_no, int Mode, double L, double step_size, bool is_store_weight, bool is_debug_mode, bool is_store_result) {
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
                    throw std::string("400 Unrecognized Mode.");
                    break;
            }
            // INNER_LOOP
            for(size_t j = 0; j < inner_m ; j ++) {
                int rand_samp = distribution(generator);
                double inner_core = model->first_component_oracle_core_dense(X, Y, N
                    , rand_samp, inner_weights);
                for(size_t k = 0; k < MAX_DIM; k ++) {
                    double val = X[rand_samp * MAX_DIM + k];
                    double vr_sub_grad = (inner_core - full_grad_core[rand_samp]) * val
                             + inner_weights[k]* lambda[0] + full_grad[k];
                    inner_weights[k] -= step_size * vr_sub_grad;
                    aver_weights[k] += inner_weights[k] / inner_m;
                }
                total_iterations ++;
                // For Drawing
                // if(is_store_weight) {
                //     for(size_t k = 0; k < MAX_DIM; k ++)
                //         stored_weights->push_back(inner_weights[k]);
                // }
                // if(is_debug_mode) {
                //     double log_F = log(model->zero_oracle_dense(X, Y, N, inner_weights));
                //     printf("SVRG: Outter Iteration: %zd -> Inner Iteration %zd, log_F for inner_weights: %lf.\n", i, j, log_F);
                // }
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
            // if(is_debug_mode) {
            //     double log_F = log(model->zero_oracle_dense(X, Y, N));
            //     printf("SVRG: Outter Iteration %zd, log_F: %lf.\n", i, log_F);
            // }
        }
        delete[] full_grad;
        delete[] inner_weights;
        //Final Output
        // double log_F = log(model->zero_oracle_dense(X, Y, N));

        // printf("SVRG: Total Iteration No.: %zd, logF = %lf.\n", total_iterations, log_F);
        // if(is_store_weight)
        //     return stored_weights;
        if(is_store_result)
            return stored_F;
        return NULL;
}

// Only Applicable for L2 regularizer
std::vector<double>* grad_desc_dense::Ada_SVRG(double* X, double* Y, size_t N, blackbox* model
    , size_t iteration_no, int Mode, double L, double step_size, bool is_store_weight, bool is_debug_mode, bool is_store_result) {
        // Random Generator
        std::random_device rd;
        std::default_random_engine generator(rd());
        std::vector<double>* stored_F = new std::vector<double>;
        double* SVRG_weights = new double[MAX_DIM];
        double* full_grad = new double[MAX_DIM];

        size_t m0 = 400;
        size_t n = m0;
        size_t loop_no = 0;
        double c = 0.000001 / (1.0 / sqrt((double) N));
        double grad_norm = 0;
        while(n < N) {
            copy_vec(SVRG_weights, model->get_model());
            if(loop_no != 0)
                n = (2 * n <= N) ? 2 * n : N;
            double Vn = 1.0 / sqrt((double) n);
            double lambda = c * Vn;
            double step_size_fixed = 0.1 / (L + c * Vn);
            std::uniform_int_distribution<int> distribution(0, n - 1);

            // Comp grad
            memset(full_grad, 0, MAX_DIM * sizeof(double));
            for(size_t j = 0; j < n; j ++) {
                double core = model->first_component_oracle_core_dense(X, Y, N, j);
                for(size_t k = 0; k < MAX_DIM; k ++) {
                    full_grad[k] += (X[j * MAX_DIM + k] * core) / (double) n;
                }
            }
            for(size_t k = 0; k < MAX_DIM; k ++)
                full_grad[k] += c * Vn * SVRG_weights[k];
            grad_norm = comp_l2_norm(full_grad);
            // OUTTER_LOOP
            while(grad_norm > sqrt((double) 2 * c) * Vn) {
                double* full_grad_core = new double[n];
                double inner_m = n;
                memset(full_grad, 0, MAX_DIM * sizeof(double));
                // Full Gradient
                for(size_t j = 0; j < n; j ++) {
                    full_grad_core[j] = model->first_component_oracle_core_dense(X, Y, N, j);
                    for(size_t k = 0; k < MAX_DIM; k ++) {
                        full_grad[k] += (X[j * MAX_DIM + k] * full_grad_core[j]) / (double) n;
                    }
                }
                // INNER_LOOP
                for(size_t j = 0; j < inner_m ; j ++) {
                    int rand_samp = distribution(generator);
                    double inner_core = model->first_component_oracle_core_dense(X, Y, N
                        , rand_samp, SVRG_weights);
                    for(size_t k = 0; k < MAX_DIM; k ++) {
                        double val = X[rand_samp * MAX_DIM + k];
                        double vr_sub_grad = (inner_core - full_grad_core[rand_samp]) * val
                                 + SVRG_weights[k]* lambda + full_grad[k];
                        SVRG_weights[k] -= step_size_fixed * vr_sub_grad;
                    }
                // INNER_LOOP END
                }
                model->update_model(SVRG_weights);
                // Comp grad
                memset(full_grad, 0, MAX_DIM * sizeof(double));
                for(size_t j = 0; j < n; j ++) {
                    double core = model->first_component_oracle_core_dense(X, Y, N, j);
                    for(size_t k = 0; k < MAX_DIM; k ++) {
                        full_grad[k] += (X[j * MAX_DIM + k] * core) / (double) n;
                    }
                }
                for(size_t k = 0; k < MAX_DIM; k ++)
                    full_grad[k] += c * Vn * SVRG_weights[k];
                grad_norm = comp_l2_norm(full_grad);
                delete[] full_grad_core;
            // OUTTER_LOOP END
            }
            loop_no ++;
        }
        //printf("Final: %lf\n", log(model->zero_oracle_dense(X, Y, N)));
        // For Matlab
        if(is_store_result) {
            stored_F->push_back(model->zero_oracle_dense(X, Y, N));
        }
        delete[] full_grad;
        delete[] SVRG_weights;

        if(is_store_result)
            return stored_F;
        return NULL;
}

std::vector<double>* grad_desc_dense::Katyusha(double* X, double* Y, size_t N, blackbox* model
    , size_t iteration_no, double L, double sigma, double step_size, bool is_store_weight, bool is_debug_mode, bool is_store_result) {
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
        // lazy update extra param
        double* last_seen = new double[MAX_DIM];
        memset(full_grad, 0, MAX_DIM * sizeof(double));
        memset(aver_weights, 0, MAX_DIM * sizeof(double));
        for(size_t i = 0; i < MAX_DIM; i ++) last_seen[i] = -1;
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
                z[k] -= alpha * katyusha_grad;
                regularizer::proximal_operator(regular, z[k], alpha, lambda);
                y[k] = inner_weights[k] - step_size_y * katyusha_grad;
                regularizer::proximal_operator(regular, y[k], step_size_y, lambda);
                aver_weights[k] += compos_pow[j] / compos_base * y[k];
                // (j + 1)th Inner Iteration
                if(j < m - 1)
                    inner_weights[k] = tau_1 * z[k] + tau_2 * outter_weights[k]
                                     + (1 - tau_1 - tau_2) * y[k];
            }
            total_iterations ++;
            // if(is_debug_mode) {
            //     double log_F = log(model->zero_oracle_sparse(X, Y, Jc, Ir, N, inner_weights));
            //     printf("Katyusha: Outter Iteration: %zd -> Inner Iteration %zd, log_F for inner_weights: %lf.\n", i, j, log_F);
            // }
        }
        model->update_model(aver_weights);
        delete[] aver_weights;
        delete[] full_grad_core;
        delete[] last_seen;
        // For Matlab
        if(is_store_result) {
            stored_F->push_back(model->zero_oracle_dense(X, Y, N));
        }
        // if(is_debug_mode) {
        //     double log_F = log(model->zero_oracle_dense(X, Y, N));
        //     printf("Katyusha: Outter Iteration %zd, log_F: %lf.\n", i, log_F);
        // }
    }
    delete[] y;
    delete[] z;
    delete[] inner_weights;
    delete[] full_grad;
    delete[] compos_pow;
    //Final Output
    // double log_F = log(model->zero_oracle_dense(X, Y, N));
    // printf("Katyusha: Total Iteration No.: %zd, log_F: %lf.\n", total_iterations, log_F);
    if(is_store_result)
        return stored_F;
    return NULL;
}

std::vector<double>* grad_desc_dense::SAGA(double* X, double* Y, size_t N, blackbox* model, size_t iteration_no
    , double L, double step_size, bool is_store_weight, bool is_debug_mode, bool is_store_result) {
    // Random Generator
    std::random_device rd;
    std::default_random_engine generator(rd());
    std::uniform_int_distribution<int> distribution(0, N - 1);
    std::vector<double>* stored_F = new std::vector<double>;
    int regular = model->get_regularizer();
    double* lambda = model->get_params();
    // For Matlab
    if(is_store_result) {
        stored_F->push_back(model->zero_oracle_dense(X, Y, N));
        // Extra Pass for Create Gradient Table
        stored_F->push_back((*stored_F)[0]);
    }
    double* new_weights = new double[MAX_DIM];
    double* grad_core_table = new double[N];
    double* aver_grad = new double[MAX_DIM];
    copy_vec(new_weights, model->get_model());
    memset(aver_grad, 0, MAX_DIM * sizeof(double));
    // Init Gradient Core Table
    for(size_t i = 0; i < N; i ++) {
        grad_core_table[i] = model->first_component_oracle_core_dense(X, Y, N, i);
        for(size_t j = 0; j < MAX_DIM; j ++)
            aver_grad[j] += grad_core_table[i] * X[i * MAX_DIM + j] / N;
    }

    size_t skip_pass = 0;
    for(size_t i = 0; i < iteration_no; i ++) {
        int rand_samp = distribution(generator);
        double core = model->first_component_oracle_core_dense(X, Y, N, rand_samp, new_weights);
        double past_grad_core = grad_core_table[rand_samp];
        grad_core_table[rand_samp] = core;
        for(size_t j = 0; j < MAX_DIM; j ++) {
            // Update Weight
            new_weights[j] -= step_size * ((core - past_grad_core)* X[rand_samp * MAX_DIM + j]
                            + aver_grad[j]);
            // Update Gradient Table Average
            aver_grad[j] -= (past_grad_core - core) * X[rand_samp * MAX_DIM + j] / N;
            regularizer::proximal_operator(regular, new_weights[j], step_size, lambda);
        }
        // For Matlab
        if(is_store_result) {
            if(!(i % N)) {
                skip_pass ++;
                if(skip_pass == 3) {
                    stored_F->push_back(model->zero_oracle_dense(X, Y, N, new_weights));
                    skip_pass = 0;
                }
            }
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
