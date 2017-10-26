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

std::vector<double>* grad_desc_async_sparse::ASAGA_Single(double* X, double* Y, size_t* Jc, size_t* Ir, size_t N
    , blackbox* model, size_t iteration_no, double L, double step_size, bool is_store_result) {
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
    double* x = new double[MAX_DIM];
    double* grad_core_table = new double[N];
    double* aver_grad = new double[MAX_DIM];
    // "Anticipate" Update Extra parameters
    double* reweight_diag = new double[MAX_DIM];
    memset(aver_grad, 0, MAX_DIM * sizeof(double));
    memset(reweight_diag, 0, MAX_DIM * sizeof(double));
    copy_vec(x, model->get_model());
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

    for(size_t i = 0; i < iteration_no; i ++) {
        int rand_samp = distribution(generator);
        double core = model->first_component_oracle_core_sparse(X, Y, Jc, Ir, N, rand_samp, x);
        double past_grad_core = grad_core_table[rand_samp];
        grad_core_table[rand_samp] = core;
        for(size_t j = Jc[rand_samp]; j < Jc[rand_samp + 1]; j ++) {
            size_t index = Ir[j];
            // Update Weight (Using Unbiased Sparse Estimate of Aver_grad)
            x[index] -= step_size * ((core - past_grad_core)* X[j]
                                + reweight_diag[index] * aver_grad[index]);
            // Update Gradient Table Average
            aver_grad[index] -= (past_grad_core - core) * X[j] / N;
            // Re-Weighted Sparse Estimate of regularizer
            regularizer::proximal_operator(regular, x[index], reweight_diag[index] * step_size, lambda);
        }
        // For Matlab
        if(is_store_result) {
            if(!((i + 1) % (3 * N))) {
                stored_F->push_back(model->zero_oracle_sparse(X, Y, Jc, Ir, N, x));
            }
        }
    }
    model->update_model(x);
    delete[] x;
    delete[] grad_core_table;
    delete[] aver_grad;
    delete[] reweight_diag;
    // For Matlab
    if(is_store_result)
        return stored_F;
    return NULL;
}

std::atomic<int> pass_counter(0);
void grad_desc_async_sparse::ASAGA_Async_Loop(double* X, double* Y, size_t* Jc, size_t* Ir, size_t N
    , std::atomic<double>* x, blackbox* model, size_t inner_iters, double step_size
    , double* reweight_diag, std::atomic<double>* grad_core_table, std::atomic<double>* aver_grad
    , std::vector<double>* stored_F, bool is_store_result) {
    // Random Generator
    std::random_device rd;
    std::default_random_engine generator(rd());
    std::uniform_int_distribution<int> distribution(0, N - 1);
    int regular = model->get_regularizer();
    int iter_no;
    double* lambda = model->get_params();
    double* inconsis_x = new double[MAX_DIM];
    for(size_t i = 0; i < inner_iters; i ++) {
        int rand_samp = distribution(generator);
        // Inconsistant Read [X] [a].
        for(size_t k = Jc[rand_samp]; k < Jc[rand_samp + 1]; k ++) {
            inconsis_x[Ir[k]] = x[Ir[k]];
        }
        iter_no = pass_counter.fetch_add(1);
        double inconsis_grad_core = grad_core_table[rand_samp];
        double core = model->first_component_oracle_core_sparse(X, Y, Jc, Ir, N, rand_samp, inconsis_x);
        double incr_grad_core = core - inconsis_grad_core;
        fetch_n_add_atomic(grad_core_table[rand_samp], incr_grad_core);
        for(size_t j = Jc[rand_samp]; j < Jc[rand_samp + 1]; j ++) {
            size_t index = Ir[j];
            // Update Weight (Using Unbiased Sparse Estimate of Aver_grad)
            double temp_x = inconsis_x[index] -  step_size * (incr_grad_core * X[j]
                        + reweight_diag[index] * aver_grad[index]);
            // Re-Weighted Sparse Estimate of regularizer
            double incr_x = regularizer::proximal_operator(regular, temp_x, reweight_diag[index] * step_size, lambda)
                             - inconsis_x[index];
            // Atomic Write
            fetch_n_add_atomic(x[index], incr_x);
            fetch_n_add_atomic(aver_grad[index], incr_grad_core * X[j] / N);
        }
        // For Matlab
        // FIXME: Whether to LOCK storing
        if(is_store_result) {
            if(!((iter_no + 1) % (3 * N))) {
                stored_F->push_back(model->zero_oracle_sparse(X, Y, Jc, Ir, N, (double *)x));
            }
        }
    }
    delete[] inconsis_x;
}

std::vector<double>* grad_desc_async_sparse::ASAGA_Async(double* X, double* Y, size_t* Jc, size_t* Ir, size_t N
    , blackbox* model, size_t iteration_no, size_t thread_no, double L, double step_size, bool is_store_result) {
    std::vector<double>* stored_F = new std::vector<double>;
    // For Matlab
    if(is_store_result) {
        stored_F->push_back(model->zero_oracle_sparse(X, Y, Jc, Ir, N));
        // Extra Pass for Create Gradient Table
        stored_F->push_back((*stored_F)[0]);
    }
    std::atomic<double>* x = new std::atomic<double>[MAX_DIM];
    std::atomic<double>* grad_core_table = new std::atomic<double>[N];
    std::atomic<double>* aver_grad = new std::atomic<double>[MAX_DIM];
    // "Anticipate" Update Extra parameters
    double* reweight_diag = new double[MAX_DIM];
    memset(aver_grad, 0, MAX_DIM * sizeof(double));
    memset(reweight_diag, 0, MAX_DIM * sizeof(double));
    copy_vec((double *)x, model->get_model());
    // Init Gradient Core Table
    for(size_t i = 0; i < N; i ++) {
        grad_core_table[i] = model->first_component_oracle_core_sparse(X, Y, Jc, Ir, N, i);
        for(size_t j = Jc[i]; j < Jc[i + 1]; j ++) {
            aver_grad[Ir[j]] = aver_grad[Ir[j]] + grad_core_table[i] * X[j] / N;
            // Compute Re-weight Matrix(Inversed)
            reweight_diag[Ir[j]] += 1.0 / (double) N;
        }
    }
    // Compute Re-weight Matrix
    for(size_t i = 0; i < MAX_DIM; i ++)
        reweight_diag[i] = 1.0 / reweight_diag[i];
    // Parallel INNER_LOOP
    std::vector<std::thread> thread_pool;
    for(size_t k = 1; k <= thread_no; k ++) {
        size_t inner_iters;
        if(k == 1)
            inner_iters = iteration_no - iteration_no / thread_no * thread_no
                + iteration_no / thread_no;
        else
            inner_iters = iteration_no / thread_no;

        thread_pool.push_back(std::thread(ASAGA_Async_Loop, X, Y, Jc, Ir, N
            , x, model, inner_iters, step_size, reweight_diag, grad_core_table
            , aver_grad, stored_F, is_store_result));
    }
    for(auto& t : thread_pool)
        t.join();
    model->update_model((double *)x);
    delete[] x;
    delete[] grad_core_table;
    delete[] aver_grad;
    delete[] reweight_diag;
    // For Matlab
    if(is_store_result)
        return stored_F;
    return NULL;
}

std::vector<double>* grad_desc_async_sparse::ASAGA(double* X, double* Y, size_t* Jc, size_t* Ir, size_t N
    , blackbox* model, size_t iteration_no, size_t thread_no, double L, double step_size, bool is_store_result) {
    if(thread_no == 1) {
        return ASAGA_Single(X, Y, Jc, Ir, N, model, iteration_no, L, step_size, is_store_result);
    }
    else {
        pass_counter = 0;
        return ASAGA_Async(X, Y, Jc, Ir, N, model, iteration_no, thread_no, L, step_size, is_store_result);
    }
}
