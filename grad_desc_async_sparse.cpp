#include "grad_desc_async_sparse.hpp"
#include "utils.hpp"
#include "regularizer.hpp"
#include <atomic>
#include <random>
#include <cmath>
#include <thread>
#include <mutex>
#include <string.h>

extern size_t MAX_DIM;

void grad_desc_async_sparse::Partial_Gradient(double* full_grad_core, size_t thread_no
    , double* X, double* Y, size_t* Jc, size_t* Ir, std::atomic<double>* full_grad
    , size_t N, blackbox* model, size_t _thread, double* _weights, std::atomic<double>* reweight_diag) {
    double* _pf = new double[MAX_DIM];
    double* _prd = new double[MAX_DIM];
    memset(_pf, 0, MAX_DIM * sizeof(double));
    memset(_prd, 0, MAX_DIM * sizeof(double));
    for(size_t i = ceil((double) N / thread_no * (_thread - 1));
            i < ceil((double) N / thread_no * _thread);
            i ++) {
        full_grad_core[i] = model->first_component_oracle_core_sparse(X, Y, Jc, Ir, N, i, _weights);
        for(size_t j = Jc[i]; j < Jc[i + 1]; j ++) {
            _pf[Ir[j]] += X[j] * full_grad_core[i] / (double) N;
            // Compute Re-weight Matrix(Inversed) in First Pass
            if(reweight_diag != NULL)
                _prd[Ir[j]] += 1.0 / (double) N;
        }
    }
    // Atomic Write
    for(size_t i = 0; i < MAX_DIM; i ++) {
        fetch_n_add_atomic(full_grad[i], _pf[i]);
        if(reweight_diag != NULL)
            fetch_n_add_atomic(reweight_diag[i], _prd[i]);
    }
    delete[] _prd;
    delete[] _pf;
}

double* grad_desc_async_sparse::Comp_Full_Grad_Parallel(double* full_grad_core, size_t thread_no
    , double* X, double* Y, size_t* Jc, size_t* Ir, size_t N, blackbox* model, double* _weights
    , std::atomic<double>* reweight_diag) {
    // Thread Pool
    std::vector<std::thread> thread_pool;
    std::atomic<double>* full_grad = new std::atomic<double>[MAX_DIM];
    memset(full_grad, 0, MAX_DIM * sizeof(double));
    for(size_t i = 1; i <= thread_no; i ++) {
        thread_pool.push_back(std::thread(Partial_Gradient, full_grad_core, thread_no
            , X, Y, Jc, Ir, full_grad, N, model, i, _weights, reweight_diag));
    }
    for(auto &t : thread_pool)
        t.join();
    double* full_grad_n = new double[MAX_DIM];
    for(size_t i = 0; i < MAX_DIM; i ++)
        full_grad_n[i] = full_grad[i];
    delete[] full_grad;
    return full_grad_n;
}

// Prox_SVRG
std::vector<double>* grad_desc_async_sparse::Prox_ASVRG_Single(double* X, double* Y, size_t* Jc, size_t* Ir
    , size_t N, blackbox* model, size_t iteration_no, int Mode, double L, double step_size
    , bool is_store_result) {
    // Random Generator
    std::random_device rd;
    std::default_random_engine generator(rd());
    std::uniform_int_distribution<int> distribution(0, N - 1);
    std::vector<double>* stored_F = new std::vector<double>;
    double* inner_weights = new double[MAX_DIM];
    double* full_grad = new double[MAX_DIM];
    // "Anticipate" Update Extra parameters
    double* reweight_diag = new double[MAX_DIM];
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
        double inner_m = m0;
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
                break;
            case SVRG_AVER_LAST:
                copy_vec(aver_weights, inner_weights);
                break;
            case SVRG_AVER_AVER:
                copy_vec(inner_weights, model->get_model());
                copy_vec(aver_weights, model->get_model());
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
                double prev_x = inner_weights[index];
                inner_weights[index] -= step_size * vr_sub_grad;
                regularizer::proximal_operator(regular, inner_weights[index]
                        , reweight_diag[index] * step_size, lambda);
                aver_weights[index] += (inner_weights[index] - prev_x) * (inner_m - j) / inner_m;
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

std::atomic<int> SVRG_counter(0);
void grad_desc_async_sparse::Prox_ASVRG_Async_Inner_Loop(double* X, double* Y, size_t* Jc
    , size_t* Ir, size_t N, std::atomic<double>* x, std::atomic<double>* aver_x
    , blackbox* model, size_t m, size_t inner_iters, double step_size, std::atomic<double>* reweight_diag
    , double* full_grad_core, double* full_grad) {
    // Random Generator
    std::random_device rd;
    std::default_random_engine generator(rd());
    std::uniform_int_distribution<int> distribution(0, N - 1);

    int regular = model->get_regularizer();
    int iter_no;
    double* lambda = model->get_params();
    double* inconsis_x = new double[MAX_DIM];
    for(size_t j = 0; j < inner_iters; j ++) {
        int rand_samp = distribution(generator);
        // Inconsistant Read [X].
        for(size_t k = Jc[rand_samp]; k < Jc[rand_samp + 1]; k ++)
            inconsis_x[Ir[k]] = x[Ir[k]];
        iter_no = SVRG_counter.fetch_add(1);
        double inner_core = model->first_component_oracle_core_sparse(X, Y
                    , Jc, Ir, N, rand_samp, inconsis_x);
        for(size_t k = Jc[rand_samp]; k < Jc[rand_samp + 1]; k ++) {
            size_t index = Ir[k];
            double val = X[k];
            double vr_sub_grad = (inner_core - full_grad_core[rand_samp]) * val
                        + reweight_diag[index] * full_grad[index];
            double temp_x = inconsis_x[index] - step_size * vr_sub_grad;
            double incr_x = regularizer::proximal_operator(regular, temp_x
                    , reweight_diag[index] * step_size, lambda) - inconsis_x[index];
            // Atomic Write
            fetch_n_add_atomic(x[index], incr_x);
            fetch_n_add_atomic(aver_x[index], incr_x * (m - iter_no) / m);
        }
    }
    delete[] inconsis_x;
}

std::vector<double>* grad_desc_async_sparse::Prox_ASVRG_Async(double* X, double* Y, size_t* Jc, size_t* Ir
    , size_t N, blackbox* model, size_t iteration_no, size_t thread_no, int Mode, double L, double step_size
    , bool is_store_result) {
    std::vector<double>* stored_F = new std::vector<double>;
    std::atomic<double>* x = new std::atomic<double>[MAX_DIM];
    // "Anticipate" Update Extra parameters
    std::atomic<double>* reweight_diag = new std::atomic<double>[MAX_DIM];
    // Average Iterates
    std::atomic<double>* aver_x = new std::atomic<double>[MAX_DIM];
    size_t m = N * 2;
    memset(reweight_diag, 0, MAX_DIM * sizeof(double));
    copy_vec((double *)x, model->get_model());
    // Init Weight Evaluate
    if(is_store_result)
        stored_F->push_back(model->zero_oracle_sparse(X, Y, Jc, Ir, N));
    // OUTTER_LOOP
    for(size_t i = 0 ; i < iteration_no; i ++) {
        double* outter_x = model->get_model();
        double* full_grad_core = new double[N];
        double* full_grad;
        SVRG_counter = 0;
        // Full Gradient
        if(i == 0) {
            full_grad = Comp_Full_Grad_Parallel(full_grad_core, thread_no
                , X, Y, Jc, Ir, N, model, outter_x, reweight_diag);
            // Compute Re-weight Matrix in First Pass
            for(size_t j = 0; j < MAX_DIM; j ++)
                reweight_diag[j] = 1.0 / reweight_diag[j];
        }
        else
            full_grad = Comp_Full_Grad_Parallel(full_grad_core, thread_no
                , X, Y, Jc, Ir, N, model, outter_x);

        switch(Mode) {
            case SVRG_LAST_LAST:
                break;
            case SVRG_AVER_LAST:
                copy_vec((double *)aver_x, (double *)x);
                break;
            case SVRG_AVER_AVER:
                copy_vec((double *)x, (double *)outter_x);
                copy_vec((double *)aver_x, (double *)outter_x);
                break;
            default:
                throw std::string("400 Unrecognized Mode.");
                break;
        }

        // Parallel INNER_LOOP
        std::vector<std::thread> thread_pool;
        for(size_t k = 1; k <= thread_no; k ++) {
            size_t inner_iters;
            if(k == 1)
                inner_iters = m - m / thread_no * thread_no + m / thread_no;
            else
                inner_iters = m / thread_no;

            thread_pool.push_back(std::thread(Prox_ASVRG_Async_Inner_Loop, X, Y, Jc, Ir, N
                , x, aver_x, model, m, inner_iters, step_size, reweight_diag, full_grad_core
                , full_grad));
        }
        for(auto& t : thread_pool)
            t.join();

        switch(Mode) {
            case SVRG_LAST_LAST:
                model->update_model((double*) x);
                break;
            case SVRG_AVER_LAST:
            case SVRG_AVER_AVER:
                model->update_model((double*) aver_x);
                break;
            default:
                throw std::string("500 Internal Error.");
                break;
        }
        // For Matlab (per m/n passes)
        if(is_store_result) {
            stored_F->push_back(model->zero_oracle_sparse(X, Y, Jc, Ir, N));
        }
        delete[] full_grad_core;
        delete[] full_grad;
    }
    delete[] reweight_diag;
    delete[] x;
    delete[] aver_x;
    if(is_store_result)
        return stored_F;
    return NULL;
}

std::vector<double>* grad_desc_async_sparse::Prox_ASVRG(double* X, double* Y, size_t* Jc, size_t* Ir
    , size_t N, blackbox* model, size_t iteration_no, size_t thread_no, int Mode, double L, double step_size
    , bool is_store_result) {
    if(thread_no == 1) {
        return Prox_ASVRG_Single(X, Y, Jc, Ir, N, model, iteration_no, Mode, L
                    , step_size, is_store_result);
    }
    else {
        return Prox_ASVRG_Async(X, Y, Jc, Ir, N, model, iteration_no, thread_no, Mode, L
                    , step_size, is_store_result);
    }
}

// Katyusha
std::vector<double>* grad_desc_async_sparse::A_Katyusha(double* X, double* Y, size_t* Jc, size_t* Ir
    , size_t N, blackbox* model, size_t iteration_no, size_t thread_no, double L, double sigma
    , double step_size, bool is_store_result) {
    if(thread_no == 1) {
        return A_Katyusha_Single(X, Y, Jc, Ir, N, model, iteration_no, L, sigma
            , step_size, is_store_result);
    }
    else {
        return A_Katyusha_Async(X, Y, Jc, Ir, N, model, iteration_no, thread_no
            , L, sigma, step_size, is_store_result);
    }
}

std::vector<double>* grad_desc_async_sparse::A_Katyusha_Single(double* X, double* Y
    , size_t* Jc, size_t* Ir, size_t N, blackbox* model, size_t iteration_no, double L
    , double sigma, double step_size, bool is_store_result) {
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
    double* x = new double[MAX_DIM];
    double* full_grad = new double[MAX_DIM];
    // "Anticipate" Update Extra parameters
    double* reweight_diag = new double[MAX_DIM];
    // init vectors
    copy_vec(y, model->get_model());
    copy_vec(z, model->get_model());
    copy_vec(x, model->get_model());
    memset(reweight_diag, 0, MAX_DIM * sizeof(double));
    // Init Weight Evaluate
    if(is_store_result)
        stored_F->push_back(model->zero_oracle_sparse(X, Y, Jc, Ir, N));
    // OUTTER LOOP
    for(size_t i = 0; i < iteration_no; i ++) {
        double* full_grad_core = new double[N];
        double* outter_x = (model->get_model());
        double* aver_y = new double[MAX_DIM];
        memset(full_grad, 0, MAX_DIM * sizeof(double));
        copy_vec(aver_y, y);
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
            x[k] = tau_1 * z[k] + tau_2 * outter_x[k]
                             + (1 - tau_1 - tau_2) * y[k];
        // INNER LOOP
        for(size_t j = 0; j < m; j ++) {
            int rand_samp = distribution(generator);
            double inner_core = model->first_component_oracle_core_sparse(X, Y, Jc, Ir, N, rand_samp, x);
            for(size_t k = Jc[rand_samp]; k < Jc[rand_samp + 1]; k ++) {
                size_t index = Ir[k];
                double val = X[k];
                double katyusha_grad = reweight_diag[index] * full_grad[index] + val * (inner_core - full_grad_core[rand_samp]);
                double prev_y = y[index], prev_z = z[index];
                z[index] -= alpha * katyusha_grad;
                regularizer::proximal_operator(regular, z[index], reweight_diag[index] * alpha, lambda);

                ////// For Katyusha With Update Option I //////
                y[index] = x[index] - step_size_y * katyusha_grad;
                regularizer::proximal_operator(regular, y[index], reweight_diag[index] * step_size_y, lambda);

                ////// For Katyusha With Update Option II //////
                // y[index] = x[index] + tau_1 * (z[index] - prev_z);
                switch(regular) {
                    case regularizer::L2:
                    case regularizer::ELASTIC_NET: // Strongly Convex Case
                        aver_y[index] += (y[index] - prev_y)
                                * equal_ratio2(compos_pow[j], compos_factor, compos_pow[m - j], m - j) / compos_base;
                        break;
                    case regularizer::L1: // Non-Strongly Convex Case
                        aver_y[index] += (y[index] - prev_y) * (m - j) / m;
                        break;
                    default:
                        throw std::string("500 Internal Error.");
                        break;
                }
                // (j + 1)th Inner Iteration
                if(j < m - 1)
                    x[index] = tau_1 * z[index] + tau_2 * outter_x[index]
                                     + (1 - tau_1 - tau_2) * y[index];
            }
            total_iterations ++;
        }

        model->update_model(aver_y);
        delete[] aver_y;
        delete[] full_grad_core;
        // For Matlab
        if(is_store_result) {
            stored_F->push_back(model->zero_oracle_sparse(X, Y, Jc, Ir, N));
        }
    }
    delete[] x;
    delete[] y;
    delete[] z;
    delete[] full_grad;
    delete[] compos_pow;
    delete[] reweight_diag;
    if(is_store_result)
        return stored_F;
    return NULL;
}

std::mutex katyusha_mutex;
std::atomic<int> Katyu_counter(0);
void grad_desc_async_sparse::A_Katyusha_Async_Inner_Loop(double* X, double* Y
    , size_t* Jc, size_t* Ir, size_t N, std::atomic<double>* y, std::atomic<double>* z
    , std::atomic<double>* aver_y, blackbox* model, size_t m, size_t inner_iters
    , double step_size_y, double tau_1, double tau_2, double alpha, double compos_factor
    , double compos_base, std::atomic<double>* reweight_diag, double* full_grad_core
    , double* full_grad, double* compos_pow, double* outter_x) {
    // Random Generator
    std::random_device rd;
    std::default_random_engine generator(rd());
    std::uniform_int_distribution<int> distribution(0, N - 1);
    int regular = model->get_regularizer();
    int iter_no;
    double* lambda = model->get_params();
    double* incr_y = new double[MAX_DIM];
    double* incr_z = new double[MAX_DIM];
    double* inconsis_x = new double[MAX_DIM];
    double* inconsis_y = new double[MAX_DIM];
    double* inconsis_z = new double[MAX_DIM];
    for(size_t j = 0; j < inner_iters; j ++) {
        int rand_samp = distribution(generator);
        // Inconsistant Read X.
        for(size_t k = Jc[rand_samp]; k < Jc[rand_samp + 1]; k ++) {
            inconsis_y[Ir[k]] = y[Ir[k]];
            inconsis_z[Ir[k]] = z[Ir[k]];
            inconsis_x[Ir[k]] = tau_1 * inconsis_z[Ir[k]] + tau_2 * outter_x[Ir[k]]
                + (1.0 - tau_1 - tau_2) * inconsis_y[Ir[k]];
        }
        iter_no = Katyu_counter.fetch_add(1);
        double inner_core = model->first_component_oracle_core_sparse(X, Y, Jc, Ir, N, rand_samp, inconsis_x);
        for(size_t k = Jc[rand_samp]; k < Jc[rand_samp + 1]; k ++) {
            size_t index = Ir[k];
            double val = X[k];
            double katyusha_grad = reweight_diag[index] * full_grad[index]
                                + val * (inner_core - full_grad_core[rand_samp]);
            // Save Increments
            double temp_z = inconsis_z[index] - alpha * katyusha_grad;
            incr_z[index] = regularizer::proximal_operator(regular, temp_z, reweight_diag[index] * alpha, lambda)
                    - inconsis_z[index];
            ////// For Katyusha With Update Option I //////
            double temp_y = inconsis_x[index] - step_size_y * katyusha_grad;
            incr_y[index] = regularizer::proximal_operator(regular, temp_y, reweight_diag[index] * step_size_y, lambda)
                - inconsis_y[index];

            ////// For Katyusha With Update Option II //////
            // incr_y[index] = inconsis_x[index] + tau_1 * incr_z[index] - inconsis_y[index];
        }

        // Lock Write
        std::lock_guard<std::mutex> lock(katyusha_mutex);
        // Atomic Write
        for(size_t k = Jc[rand_samp]; k < Jc[rand_samp + 1]; k ++) {
            size_t index = Ir[k];
            fetch_n_add_atomic(z[index], incr_z[index]);
            fetch_n_add_atomic(y[index], incr_y[index]);
            // Update Average y
            // switch(regular) {
            //     case regularizer::L2:
            //     case regularizer::ELASTIC_NET:{ // Strongly Convex Case
            //         fetch_n_add_atomic(aver_y[index],  incr_y[index]
            //                 * equal_ratio2(compos_pow[j], compos_factor, compos_pow[m - iter_no], m - iter_no) / compos_base);
            //         break;
            //     }
            //     case regularizer::L1:{ // Non-Strongly Convex Case
            //         fetch_n_add_atomic(aver_y[index], incr_y[index] * (m - iter_no) / m);
            //         break;
            //     }
            //     default:
            //         throw std::string("500 Internal Error.");
            //         break;
            // }
        }
    }
    delete[] inconsis_x;
    delete[] inconsis_y;
    delete[] inconsis_z;
    delete[] incr_y;
    delete[] incr_z;
}

std::vector<double>* grad_desc_async_sparse::A_Katyusha_Async(double* X, double* Y
    , size_t* Jc, size_t* Ir, size_t N, blackbox* model, size_t iteration_no, size_t thread_no
    , double L, double sigma, double step_size, bool is_store_result) {
    std::vector<double>* stored_F = new std::vector<double>;
    size_t m = 2.0 * N;
    int regular = model->get_regularizer();
    double tau_2 = 0.5, tau_1 = 0.5;
    if(sqrt(sigma * m / (3.0 * L)) < 0.5) tau_1 = sqrt(sigma * m / (3.0 * L));
    double alpha = 1.0 / (tau_1 * 3.0 * L);
    double step_size_y = 1.0 / (3.0 * L);
    double compos_factor = 1.0 + alpha * sigma;
    double compos_base = (pow((double)compos_factor, (double)m) - 1.0) / (alpha * sigma);
    double* compos_pow = new double[m + 1];
    for(size_t i = 0; i <= m; i ++)
        compos_pow[i] = pow((double)compos_factor, (double)i);
    std::atomic<double>* y = new std::atomic<double>[MAX_DIM];
    std::atomic<double>* z = new std::atomic<double>[MAX_DIM];
    std::atomic<double>* aver_y = new std::atomic<double>[MAX_DIM];
    // "Anticipate" Update Extra parameters
    std::atomic<double>* reweight_diag = new std::atomic<double>[MAX_DIM];
    // init vectors
    copy_vec((double *)y, model->get_model());
    copy_vec((double *)z, model->get_model());
    memset(reweight_diag, 0, MAX_DIM * sizeof(double));
    // Init Weight Evaluate
    if(is_store_result)
        stored_F->push_back(model->zero_oracle_sparse(X, Y, Jc, Ir, N));
    // OUTTER LOOP
    for(size_t i = 0; i < iteration_no; i ++) {
        double* full_grad_core = new double[N];
        double* outter_x = (model->get_model());
        copy_vec((double *)aver_y, (double *)y);
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
        double* full_grad;
        if(i == 0) {
            full_grad = Comp_Full_Grad_Parallel(full_grad_core, thread_no
                , X, Y, Jc, Ir, N, model, outter_x, reweight_diag);
            // Compute Re-weight Matrix in First Pass
            for(size_t j = 0; j < MAX_DIM; j ++)
                reweight_diag[j] = 1.0 / reweight_diag[j];
        }
        else
            full_grad = Comp_Full_Grad_Parallel(full_grad_core, thread_no
                , X, Y, Jc, Ir, N, model, outter_x);
        Katyu_counter = 0;
        // Parallel INNER LOOP
        std::vector<std::thread> thread_pool;
        for(size_t k = 1; k <= thread_no; k ++) {
            size_t inner_iters;
            if(k == 1)
                inner_iters = m - m / thread_no * thread_no + m / thread_no;
            else
                inner_iters = m / thread_no;

            thread_pool.push_back(std::thread(A_Katyusha_Async_Inner_Loop, X, Y
                , Jc, Ir, N, y, z, aver_y, model, m, inner_iters, step_size_y
                , tau_1, tau_2, alpha, compos_factor, compos_base, reweight_diag
                , full_grad_core, full_grad, compos_pow, outter_x));
        }
        for(auto& t : thread_pool)
            t.join();
        model->update_model((double*) y);
        delete[] full_grad_core;
        delete[] full_grad;
        // For Matlab
        if(is_store_result) {
            stored_F->push_back(model->zero_oracle_sparse(X, Y, Jc, Ir, N));
        }
    }
    delete[] y;
    delete[] z;
    delete[] aver_y;
    delete[] compos_pow;
    delete[] reweight_diag;
    if(is_store_result)
        return stored_F;
    return NULL;
}

// // Only L2, TODO: Implement Real Asynchronous SVRG (AsySVRG)
// std::vector<double>* grad_desc_async_sparse::ASVRG(double* X, double* Y, size_t* Jc, size_t* Ir
//     , size_t N, blackbox* model, size_t iteration_no, size_t thread_no, int Mode, double L
//     , double step_size, bool is_store_result) {
//         // Random Generator
//         std::random_device rd;
//         std::default_random_engine generator(rd());
//         std::uniform_int_distribution<int> distribution(0, N - 1);
//         std::vector<double>* stored_F = new std::vector<double>;
//         double* inner_weights = new double[MAX_DIM];
//         double* full_grad = new double[MAX_DIM];
//         // "Anticipate" Update Extra parameters
//         double* reweight_diag = new double[MAX_DIM];
//         double* lambda = model->get_params();
//         //FIXME: Epoch Size(SVRG / SVRG++)
//         double m0 = (double) N * 2.0;
//         size_t total_iterations = 0;
//         memset(reweight_diag, 0, MAX_DIM * sizeof(double));
//         copy_vec(inner_weights, model->get_model());
//         // Init Weight Evaluate
//         if(is_store_result)
//             stored_F->push_back(model->zero_oracle_sparse(X, Y, Jc, Ir, N));
//         // OUTTER_LOOP
//         for(size_t i = 0 ; i < iteration_no; i ++) {
//             double* full_grad_core = new double[N];
//             // Average Iterates
//             double* aver_weights = new double[MAX_DIM];
//             //FIXME: SVRG / SVRG++
//             double inner_m = m0;//pow(2, i + 1) * m0;
//             memset(aver_weights, 0, MAX_DIM * sizeof(double));
//             memset(full_grad, 0, MAX_DIM * sizeof(double));
//             // Full Gradient
//             for(size_t j = 0; j < N; j ++) {
//                 full_grad_core[j] = model->first_component_oracle_core_sparse(X, Y, Jc, Ir, N, j);
//                 for(size_t k = Jc[j]; k < Jc[j + 1]; k ++) {
//                     full_grad[Ir[k]] += (X[k] * full_grad_core[j]) / (double) N;
//                     // Compute Re-weight Matrix(Inversed) in First Pass
//                     if(i == 0)
//                         reweight_diag[Ir[k]] += 1.0 / (double) N;
//                 }
//             }
//             // Compute Re-weight Matrix in First Pass
//             if(i == 0)
//                 for(size_t j = 0; j < MAX_DIM; j ++)
//                     reweight_diag[j] = 1.0 / reweight_diag[j];
//
//             switch(Mode) {
//                 case SVRG_LAST_LAST:
//                 case SVRG_AVER_LAST:
//                     break;
//                 case SVRG_AVER_AVER:
//                     copy_vec(inner_weights, model->get_model());
//                     break;
//                 default:
//                     throw std::string("400 Unrecognized Mode.");
//                     break;
//             }
//             // INNER_LOOP
//             for(size_t j = 0; j < inner_m ; j ++) {
//                 int rand_samp = distribution(generator);
//                 double inner_core = model->first_component_oracle_core_sparse(X, Y, Jc, Ir, N
//                     , rand_samp, inner_weights);
//                 for(size_t k = Jc[rand_samp]; k < Jc[rand_samp + 1]; k ++) {
//                     size_t index = Ir[k];
//                     double val = X[k];
//                     double vr_sub_grad = (inner_core - full_grad_core[rand_samp]) * val
//                              + (inner_weights[index] + model->get_model()[index] * (reweight_diag[index] - 1) ) * lambda[0]
//                              + reweight_diag[index] * full_grad[index];
//                     inner_weights[index] -= step_size * vr_sub_grad;
//                     aver_weights[index] += inner_weights[index] / inner_m;
//                 }
//                 total_iterations ++;
//             }
//             switch(Mode) {
//                 case SVRG_LAST_LAST:
//                     model->update_model(inner_weights);
//                     break;
//                 case SVRG_AVER_LAST:
//                 case SVRG_AVER_AVER:
//                     model->update_model(aver_weights);
//                     break;
//                 default:
//                     throw std::string("500 Internal Error.");
//                     break;
//             }
//             // For Matlab (per m/n passes)
//             if(is_store_result) {
//                 stored_F->push_back(model->zero_oracle_sparse(X, Y, Jc, Ir, N));
//             }
//             delete[] aver_weights;
//             delete[] full_grad_core;
//         }
//         delete[] full_grad;
//         delete[] inner_weights;
//         delete[] reweight_diag;
//         if(is_store_result)
//             return stored_F;
//         return NULL;
// }
