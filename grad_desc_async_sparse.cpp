#include "grad_desc_async_sparse.hpp"
#include "utils.hpp"
#include "regularizer.hpp"
#include <random>
#include <cmath>
#include <string.h>

extern size_t MAX_DIM;

std::vector<double>* grad_desc_async_sparse::ASAGA(double* X, double* Y, size_t* Jc, size_t* Ir, size_t N
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
    // "Anticipate" Update Extra parameters
    double* reweight_diag = new double[MAX_DIM];
    memset(aver_grad, 0, MAX_DIM * sizeof(double));
    memset(reweight_diag, 0, MAX_DIM * sizeof(double));
    copy_vec(new_weights, model->get_model());
    // Init Gradient Core Table
    for(size_t i = 0; i < N; i ++) {
        grad_core_table[i] = model->first_component_oracle_core_sparse(X, Y, Jc, Ir, N, i);
        for(size_t j = Jc[i]; j < Jc[i + 1]; j ++) {
            aver_grad[Ir[j]] += grad_core_table[i] * X[j] / N;
            // Compute Re-weight Matrix(Inversed)
            reweight_diag[Ir[j]] += 1.0 / (double) N;
        }
    }
    // Compute Re-weight Matrix
    for(size_t i = 0; i < MAX_DIM; i ++)
        reweight_diag[i] = 1.0 / reweight_diag[i];

    size_t skip_pass = 0;
    for(size_t i = 0; i < iteration_no; i ++) {
        int rand_samp = distribution(generator);
        double core = model->first_component_oracle_core_sparse(X, Y, Jc, Ir, N, rand_samp, new_weights);
        double past_grad_core = grad_core_table[rand_samp];
        grad_core_table[rand_samp] = core;
        for(size_t j = Jc[rand_samp]; j < Jc[rand_samp + 1]; j ++) {
            size_t index = Ir[j];
            // Update Weight (Using Unbiased Sparse Estimate of Aver_grad)
            new_weights[index] -= step_size * ((core - past_grad_core)* X[j]
                                + reweight_diag[index] * aver_grad[index]);
            // Update Gradient Table Average
            aver_grad[index] -= (past_grad_core - core) * X[j] / N;
            // Re-Weighted Sparse Estimate of regularizer
            regularizer::proximal_operator(regular, new_weights[index], reweight_diag[index] * step_size, lambda);
        }
        // For Matlab
        if(is_store_result) {
            if(!(i % N)) {
                skip_pass ++;
                if(skip_pass == 3) {
                    stored_F->push_back(model->zero_oracle_sparse(X, Y, Jc, Ir, N, new_weights));
                    skip_pass = 0;
                }
            }
        }
    }
    model->update_model(new_weights);
    delete[] new_weights;
    delete[] grad_core_table;
    delete[] aver_grad;
    delete[] reweight_diag;
    // For Matlab
    if(is_store_result)
        return stored_F;
    return NULL;
}

std::vector<double>* grad_desc_async_sparse::Prox_ASVRG(double* X, double* Y, size_t* Jc, size_t* Ir
    , size_t N, blackbox* model, size_t iteration_no, int Mode, double L, double step_size
    , bool is_store_weight, bool is_debug_mode, bool is_store_result) {
    // Random Generator
    std::random_device rd;
    std::default_random_engine generator(rd());
    std::uniform_int_distribution<int> distribution(0, N - 1);
    std::vector<double>* stored_F = new std::vector<double>;
    double* inner_weights = new double[MAX_DIM];
    double* full_grad = new double[MAX_DIM];
    // "Anticipate" Update Extra parameters
    double* reweight_diag = new double[MAX_DIM];
    //FIXME: Epoch Size(SVRG / SVRG++)
    double m0 = (double) N * 2.0;
    int regular = model->get_regularizer();
    double* lambda = model->get_params();
    size_t total_iterations = 0;
    memset(reweight_diag, 0, MAX_DIM * sizeof(double));
    copy_vec(inner_weights, model->get_model());
    // Init Weight Evaluate
    if(is_store_result)
        stored_F->push_back(model->zero_oracle_sparse(X, Y, Jc, Ir, N));
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
            full_grad_core[j] = model->first_component_oracle_core_sparse(X, Y, Jc, Ir, N, j);
            for(size_t k = Jc[j]; k < Jc[j + 1]; k ++) {
                full_grad[Ir[k]] += X[k] * full_grad_core[j] / (double) N;
                // Compute Re-weight Matrix(Inversed) in First Pass
                if(i == 0)
                    reweight_diag[Ir[k]] += 1.0 / (double) N;
            }
        }
        // Compute Re-weight Matrix in First Pass
        if(i == 0)
            for(size_t j = 0; j < MAX_DIM; j ++)
                reweight_diag[j] = 1.0 / reweight_diag[j];

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
                double vr_sub_grad = (inner_core - full_grad_core[rand_samp]) * val
                            + reweight_diag[index] * full_grad[index];
                inner_weights[index] -= step_size * vr_sub_grad;
                // FIXME: Average Scheme
                aver_weights[index] += regularizer::proximal_operator(regular, inner_weights[index]
                    , reweight_diag[index] * step_size, lambda) / inner_m;
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
            stored_F->push_back(model->zero_oracle_sparse(X, Y, Jc, Ir, N));
        }
        delete[] aver_weights;
        delete[] full_grad_core;
    }
    delete[] reweight_diag;
    delete[] full_grad;
    delete[] inner_weights;
    if(is_store_result)
        return stored_F;
    return NULL;
}

// Only L2, TODO: Implement Real Asynchronous SVRG (AsySVRG)
std::vector<double>* grad_desc_async_sparse::ASVRG(double* X, double* Y, size_t* Jc, size_t* Ir
    , size_t N, blackbox* model, size_t iteration_no, int Mode, double L
    , double step_size, bool is_store_weight, bool is_debug_mode, bool is_store_result) {
        // Random Generator
        std::random_device rd;
        std::default_random_engine generator(rd());
        std::uniform_int_distribution<int> distribution(0, N - 1);
        std::vector<double>* stored_F = new std::vector<double>;
        double* inner_weights = new double[MAX_DIM];
        double* full_grad = new double[MAX_DIM];
        // "Anticipate" Update Extra parameters
        double* reweight_diag = new double[MAX_DIM];
        double* lambda = model->get_params();
        //FIXME: Epoch Size(SVRG / SVRG++)
        double m0 = (double) N * 2.0;
        size_t total_iterations = 0;
        memset(reweight_diag, 0, MAX_DIM * sizeof(double));
        copy_vec(inner_weights, model->get_model());
        // Init Weight Evaluate
        if(is_store_result)
            stored_F->push_back(model->zero_oracle_sparse(X, Y, Jc, Ir, N));
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
                full_grad_core[j] = model->first_component_oracle_core_sparse(X, Y, Jc, Ir, N, j);
                for(size_t k = Jc[j]; k < Jc[j + 1]; k ++) {
                    full_grad[Ir[k]] += (X[k] * full_grad_core[j]) / (double) N;
                    // Compute Re-weight Matrix(Inversed) in First Pass
                    if(i == 0)
                        reweight_diag[Ir[k]] += 1.0 / (double) N;
                }
            }
            // Compute Re-weight Matrix in First Pass
            if(i == 0)
                for(size_t j = 0; j < MAX_DIM; j ++)
                    reweight_diag[j] = 1.0 / reweight_diag[j];

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
                    double vr_sub_grad = (inner_core - full_grad_core[rand_samp]) * val
                             + (inner_weights[index] + model->get_model()[index] * (reweight_diag[index] - 1) ) * lambda[0]
                             + reweight_diag[index] * full_grad[index];
                    inner_weights[index] -= step_size * vr_sub_grad;
                    aver_weights[index] += inner_weights[index] / inner_m;
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
                stored_F->push_back(model->zero_oracle_sparse(X, Y, Jc, Ir, N));
            }
            delete[] aver_weights;
            delete[] full_grad_core;
        }
        delete[] full_grad;
        delete[] inner_weights;
        delete[] reweight_diag;
        if(is_store_result)
            return stored_F;
        return NULL;
}

std::vector<double>* grad_desc_async_sparse::A_Katyusha(double* X, double* Y, size_t* Jc, size_t* Ir
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
    // "Anticipate" Update Extra parameters
    double* reweight_diag = new double[MAX_DIM];
    // init vectors
    copy_vec(y, model->get_model());
    copy_vec(z, model->get_model());
    copy_vec(inner_weights, model->get_model());
    memset(reweight_diag, 0, MAX_DIM * sizeof(double));
    // Init Weight Evaluate
    if(is_store_result)
        stored_F->push_back(model->zero_oracle_sparse(X, Y, Jc, Ir, N));
    // OUTTER LOOP
    for(size_t i = 0; i < iteration_no; i ++) {
        double* full_grad_core = new double[N];
        double* outter_weights = (model->get_model());
        double* aver_weights = new double[MAX_DIM];
        memset(full_grad, 0, MAX_DIM * sizeof(double));
        memset(aver_weights, 0, MAX_DIM * sizeof(double));
        // Full Gradient
        for(size_t j = 0; j < N; j ++) {
            full_grad_core[j] = model->first_component_oracle_core_sparse(X, Y, Jc, Ir, N, j);
            for(size_t k = Jc[j]; k < Jc[j + 1]; k ++) {
                full_grad[Ir[k]] += X[k] * full_grad_core[j] / (double) N;
                // Compute Re-weight Matrix(Inversed) in First Pass
                if(i == 0)
                    reweight_diag[Ir[k]] += 1.0 / (double) N;
            }
        }
        // Compute Re-weight Matrix in First Pass
        if(i == 0)
            for(size_t j = 0; j < MAX_DIM; j ++)
                reweight_diag[j] = 1.0 / reweight_diag[j];

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
                double katyusha_grad = reweight_diag[index] * full_grad[index] + val * (inner_core - full_grad_core[rand_samp]);
                z[index] -= alpha * katyusha_grad;
                regularizer::proximal_operator(regular, z[index], reweight_diag[index] * alpha, lambda);
                y[index] = inner_weights[index] - step_size_y * katyusha_grad;
                regularizer::proximal_operator(regular, y[index], reweight_diag[index] * step_size_y, lambda);
                aver_weights[index] += compos_pow[j] / compos_base * y[index];
                // (j + 1)th Inner Iteration
                if(j < m - 1)
                    inner_weights[index] = tau_1 * z[index] + tau_2 * outter_weights[index]
                                     + (1 - tau_1 - tau_2) * y[index];
            }
            total_iterations ++;
        }

        // FIXME:
        model->update_model(inner_weights);
        delete[] aver_weights;
        delete[] full_grad_core;
        // For Matlab
        if(is_store_result) {
            stored_F->push_back(model->zero_oracle_sparse(X, Y, Jc, Ir, N));
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
