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

std::atomic<int> AsyAcc1_counter(0);
void grad_desc_async_sparse::AsyAcc1_Inner_Loop(double* X, double* Y, size_t* Jc
    , size_t* Ir, size_t N, std::atomic<double>* x, std::atomic<double>* aver_x
    , blackbox* model, size_t m, size_t inner_iters, double step_size, double sigma
    , std::atomic<double>* reweight_diag, double* full_grad_core, double* full_grad
    , std::atomic<double>* moment) {
    // Random Generator
    std::random_device rd;
    std::default_random_engine generator(rd());
    std::uniform_int_distribution<int> distribution(0, N - 1);

    int regular = model->get_regularizer();
    int iter_no;
    double* lambda = model->get_params();
    double* inconsis_x = new double[MAX_DIM];
    double* virtual_y = new double[MAX_DIM];
    for(size_t j = 0; j < inner_iters; j ++) {
        int rand_samp = distribution(generator);
        // Inconsistant Read [X].
        for(size_t k = Jc[rand_samp]; k < Jc[rand_samp + 1]; k ++) {
            inconsis_x[Ir[k]] = x[Ir[k]];
            virtual_y[Ir[k]] = inconsis_x[Ir[k]] + sigma * moment[Ir[k]];
        }
        iter_no = AsyAcc1_counter.fetch_add(1);
        double inner_core = model->first_component_oracle_core_sparse(X, Y
                    , Jc, Ir, N, rand_samp, virtual_y);
        for(size_t k = Jc[rand_samp]; k < Jc[rand_samp + 1]; k ++) {
            size_t index = Ir[k];
            double val = X[k];
            double vr_sub_grad = (inner_core - full_grad_core[rand_samp]) * val
                        + reweight_diag[index] * full_grad[index];
            double temp_x = virtual_y[index] - step_size * vr_sub_grad;
            double incr_x = regularizer::proximal_operator(regular, temp_x
                    , reweight_diag[index] * step_size, lambda) - inconsis_x[index];
            // Atomic Write
            moment[index] = incr_x;
            fetch_n_add_atomic(x[index], incr_x);
            fetch_n_add_atomic(aver_x[index], incr_x * (m - iter_no) / m);
        }
    }
    delete[] virtual_y;
    delete[] inconsis_x;
}

std::vector<double>* grad_desc_async_sparse::AsyAcc1__Async(double* X, double* Y, size_t* Jc, size_t* Ir
    , size_t N, blackbox* model, size_t iteration_no, size_t thread_no, int Mode, double L, double sigma
    , double step_size, bool is_store_result) {
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
        AsyAcc1_counter = 0;
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
        std::atomic<double>* moment = new std::atomic<double>[MAX_DIM];
        memset(moment, 0, MAX_DIM * sizeof(double));
        // Parallel INNER_LOOP
        std::vector<std::thread> thread_pool;
        for(size_t k = 1; k <= thread_no; k ++) {
            size_t inner_iters;
            if(k == 1)
                inner_iters = m - m / thread_no * thread_no + m / thread_no;
            else
                inner_iters = m / thread_no;

            thread_pool.push_back(std::thread(AsyAcc1_Inner_Loop, X, Y, Jc, Ir, N
                , x, aver_x, model, m, inner_iters, step_size, sigma, reweight_diag
                , full_grad_core, full_grad, moment));
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
        delete[] moment;
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
