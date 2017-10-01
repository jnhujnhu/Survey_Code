#include "grad_desc_dense.hpp"
#include "utils.hpp"
#include "regularizer.hpp"
#include <random>
#include <cmath>
#include <string.h>

extern size_t MAX_DIM;

double* grad_desc_dense::GD(double* X, double* Y, size_t N, blackbox* model, size_t iteration_no
    , double L, double step_size, bool is_store_weight, bool is_debug_mode, bool is_store_result) {
    double* stored_F = NULL;
    size_t passes = iteration_no;
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
        //For Matlab
        if(is_store_result) {
            stored_F[i] = model->zero_oracle_dense(X, Y, N);
        }
        delete[] full_grad;
    }
    delete[] new_weights;
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
        // For Matlab
        if(is_store_result) {
            if(!((i + 1) % N)) {
                stored_F[(size_t) floor((double) i / N) + 1] = model->zero_oracle_dense(X, Y, N, new_weights);
            }
        }
    }
    model->update_model(new_weights);
    delete[] new_weights;
    // For Matlab
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

// Only Applicable for L2 regularizer
std::vector<double>* grad_desc_dense::SVRG(double* X, double* Y, size_t N, blackbox* model
    , size_t iteration_no, int Mode, double L, double step_size, bool is_store_weight, bool is_debug_mode, bool is_store_result) {
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
                for(size_t k = 0; k < MAX_DIM; k ++) {
                    double val = X[rand_samp * MAX_DIM + k];
                    double vr_sub_grad = (inner_core - full_grad_core[rand_samp]) * val
                             + inner_weights[k]* lambda[0] + full_grad[k];
                    inner_weights[k] -= step_size * vr_sub_grad;
                    aver_weights[k] += inner_weights[k] / inner_m;
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
        delete[] last_seen;
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

// Only L2. Non-Prox Update. Using Update Option I.
std::vector<double>* grad_desc_dense::Katyusha_2(double* X, double* Y, size_t N, blackbox* model
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
                double katyusha_grad = full_grad[k]
                        + val * (inner_core - full_grad_core[rand_samp]);
                z[k] -= alpha * (katyusha_grad + inner_weights[k] * lambda[0]);
                ////// Update Option I //////
                y[k] = inner_weights[k] - step_size_y * (katyusha_grad + inner_weights[k] * lambda[0]);
                aver_weights[k] += compos_pow[j] / compos_base * y[k];
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
        delete[] last_seen;
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

// Only for Ridge or Lasso
std::vector<double>* grad_desc_dense::SVRG_SD(double* X, double* Y, size_t N, blackbox* model
    , size_t iteration_no, double L, double step_size, double r, double* SV, bool is_store_weight, bool is_debug_mode, bool is_store_result) {
    // Random Generator
    std::random_device rd;
    std::default_random_engine generator(rd());
    std::uniform_int_distribution<int> distribution(0, N - 1);
    std::vector<double>* stored_F = new std::vector<double>;
    double* x = new double[MAX_DIM];
    double* y = new double[MAX_DIM];
    double* x_hat = new double[MAX_DIM];
    // Momentum Constant
    double sigma = 1.0 / 3.0;
    // Trade off parameter
    double delta = 0.1;
    double zeta = delta * step_size / (1.0 - L * step_size);
    double* full_grad = new double[MAX_DIM];
    double m0 = (double) N * 2.0;
    int regular = model->get_regularizer();
    double* lambda = model->get_params();
    size_t total_iterations = 0;
    // Compute (y^T)X
    double* yX = new double [MAX_DIM];
    memset(yX, 0, MAX_DIM * sizeof(double));
    for(size_t i = 0; i < MAX_DIM; i ++) {
        for(size_t j = 0; j < N; j ++){
            yX[i] += Y[j] * X[j * MAX_DIM + i];
        }
    }
    copy_vec(x, model->get_model());
    copy_vec(x_hat, model->get_model());
    copy_vec(y, model->get_model());
    // Init Weight Evaluate
    if(is_store_result)
        stored_F->push_back(model->zero_oracle_dense(X, Y, N));
    // OUTTER_LOOP
    for(size_t i = 0 ; i < iteration_no; i ++) {
        double* full_grad_core = new double[N];
        // Average Iterates
        double* aver_weights = new double[MAX_DIM];
        double* prev_x_hat = new double[MAX_DIM];
        double inner_m = m0;
        memset(prev_x_hat, 0, MAX_DIM * sizeof(double));
        memset(aver_weights, 0, MAX_DIM * sizeof(double));
        memset(full_grad, 0, MAX_DIM * sizeof(double));

        switch(regular) {
            case regularizer::L2:
            case regularizer::ELASTIC_NET:
                copy_vec(x, model->get_model());
                copy_vec(x_hat, model->get_model());
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
            full_grad_core[j] = model->first_component_oracle_core_dense(X, Y, N, j);
            for(size_t k = 0; k < MAX_DIM; k ++) {
                full_grad[k] += X[j * MAX_DIM + k] * full_grad_core[j] / (double) N;
            }
        }
        // INNER_LOOP
        for(size_t j = 0; j < inner_m; j ++) {
            int rand_samp = distribution(generator);
            double inner_core = model->first_component_oracle_core_dense(X, Y, N
                , rand_samp, x);
            // Compute Theta_k (every 2000 iter)
            double theta = 1.0;
            // if(!(i + 1) % 2000) {
            //     switch(regular) {
            //         case regularizer::L2: {
            //             double bAx = 0.0, square_p = 0.0, square_x = 0.0;
            //             for(size_t k = 0; k < MAX_DIM; k ++) {
            //                 bAx += yX[k] * x[k];
            //                 double difference = (inner_core - full_grad_core[rand_samp])
            //                         * X[rand_samp * MAX_DIM + k];
            //                 square_p += difference * difference;
            //                 square_x += x[k] * x[k];
            //             }
            //             bAx /= (double) N;
            //             double SVx = 0.0;
            //             for(size_t k = 0; k < r; k ++) {
            //                 double SVd = 0.0;
            //                 for(size_t l = 0; l < MAX_DIM; l ++)
            //                     SVd += SV[k * MAX_DIM + l] * x[l];
            //                 SVx += SVd * SVd;
            //             }
            //             theta = (bAx + zeta * square_p)
            //                 / (SVx / N + zeta * square_p + lambda[0] * square_x);
            //             break;
            //         }
            //         case regularizer::L1: {
            //             throw std::string("Not Done.");
            //             break;
            //         }
            //         default:
            //             throw std::string("Error");
            //             break;
            //     }
            // }
            for(size_t k = 0; k < MAX_DIM; k ++) {
                double val = X[rand_samp * MAX_DIM + k];
                double vr_sub_grad = (inner_core - full_grad_core[rand_samp]) * val + full_grad[k];
                y[k] = x[k] - step_size * vr_sub_grad;
                regularizer::proximal_operator(regular, y[k], step_size, lambda);
                // Update x
                prev_x_hat[k] = x_hat[k];
                x_hat[k] = theta * x[k];
                x[k] = y[k] + (1.0 - sigma) * (x_hat[k] - prev_x_hat[k]);
                aver_weights[k] += x_hat[k] / (double) inner_m;
            }
            total_iterations ++;
        }
        model->update_model(aver_weights);
        // NSC update y
        if(regular == regularizer::L1) {
            for(size_t k = 0; k < MAX_DIM; k ++)
                y[k] =  (x[k] - (1 - sigma) * x_hat[k]) / sigma;
        }
        // For Matlab (per m/n passes)
        if(is_store_result) {
            stored_F->push_back(model->zero_oracle_dense(X, Y, N));
        }
        delete[] aver_weights;
        delete[] full_grad_core;
        delete[] prev_x_hat;
    }
    delete[] full_grad;
    delete[] x;
    delete[] y;
    delete[] x_hat;
    if(is_store_result)
        return stored_F;
    return NULL;
}

// std::vector<double>* grad_desc_dense::SVRGA(double* X, double* Y, size_t N, blackbox* model
//     , size_t iteration_no, int Mode, double L, double step_size, bool is_store_weight, bool is_debug_mode, bool is_store_result) {
//     // Random Generator
//     std::random_device rd;
//     std::default_random_engine generator(rd());
//     std::uniform_int_distribution<int> distribution(0, N - 1);
//     // std::vector<double>* stored_weights = new std::vector<double>;
//     std::vector<double>* stored_F = new std::vector<double>;
//     double* inner_weights = new double[MAX_DIM];
//     double* full_grad = new double[MAX_DIM];
//     //FIXME: Epoch Size(SVRG / SVRG++)
//     double m0 = (double) N * 2.0;
//     int regular = model->get_regularizer();
//     double* lambda = model->get_params();
//     size_t total_iterations = 0;
//     double* grad_core_table = new double[N];
//     double* aver_grad = new double[MAX_DIM];
//     memset(aver_grad, 0, MAX_DIM * sizeof(double));
//     copy_vec(inner_weights, model->get_model());
//     // Init Weight Evaluate
//     if(is_store_result)
//         stored_F->push_back(model->zero_oracle_dense(X, Y, N));
//     // OUTTER_LOOP
//     for(size_t i = 0 ; i < iteration_no; i ++) {
//         double* full_grad_core = new double[N];
//         double* full_grad = new double[MAX_DIM];
//         double inner_m = m0;
//         memset(full_grad, 0, MAX_DIM * sizeof(double));
//         // Full Gradient
//         for(size_t j = 0; j < N; j ++) {
//             full_grad_core[j] = model->first_component_oracle_core_dense(X, Y, N, j);
//             //if(i == 0)
//                 grad_core_table[j] = full_grad_core[j];
//             for(size_t k = 0; k < MAX_DIM; k ++) {
//                 full_grad[k] += X[j * MAX_DIM + k] * full_grad_core[j] / (double) N;
//                 //if(i == 0)
//                     aver_grad[k] += grad_core_table[j] * X[j * MAX_DIM + k] / (double) N;
//             }
//         }
//
//         // INNER_LOOP
//         for(size_t j = 0; j < inner_m; j ++) {
//             int rand_samp = distribution(generator);
//             double inner_core = model->first_component_oracle_core_dense(X, Y, N
//                 , rand_samp, inner_weights);
//             double past_grad_core = grad_core_table[rand_samp];
//             grad_core_table[rand_samp] = inner_core;
//             for(size_t k = 0; k < MAX_DIM; k ++) {
//                 double val = X[rand_samp * MAX_DIM + k];
//                 double vr_sub_grad = (inner_core - 0.5 * past_grad_core - 0.5 * full_grad_core[rand_samp]) * val
//                         + 0.5 * full_grad[k] + 0.5 * aver_grad[k];
//                 inner_weights[k] -= step_size * vr_sub_grad;
//                 regularizer::proximal_operator(regular, inner_weights[k], step_size, lambda);
//                 aver_grad[k] -= (past_grad_core - inner_core) * X[rand_samp * MAX_DIM + k] / N;
//             }
//             total_iterations ++;
//         }
//         model->update_model(inner_weights);
//         // For Matlab (per m/n passes)
//         if(is_store_result) {
//             stored_F->push_back(model->zero_oracle_dense(X, Y, N));
//         }
//         delete[] full_grad_core;
//         delete[] full_grad;
//     }
//     delete[] full_grad;
//     delete[] inner_weights;
//     delete[] grad_core_table;
//     delete[] aver_grad;
//     if(is_store_result)
//         return stored_F;
//     return NULL;
// }
